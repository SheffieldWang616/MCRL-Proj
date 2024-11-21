import os
import pickle
import shutil
from argparse import ArgumentParser
import numpy as np

import torch
import torch.multiprocessing as mp
from datetime import datetime
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS
import cv2

def arg_parser():
    """
    Parse command-line arguments for the MineRL environment runner.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    
    parser = ArgumentParser("Run pretrained models on MineRL environment in parallel without stable-baselines3")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments to run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run in each environment.")
    parser.add_argument("--videolog", action='store_true', help="Enable video logging")
    parser.add_argument("--weights", type=str, default="./Model_Weights/2x_pre/rl-from-house-2x.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="./Model_Weights/2x_pre/2x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--max_frame_count", type=int, default=100, help="Number of frames to collect before saving a video.")
    return parser.parse_args()

def env_worker(model, optimizer, lock, main_conn, steps, video_log, device, short_vid_dir=None, max_frame_count=100):
    """
    Worker function to run an environment, update a shared model, and optionally record videos.
    Args:
        model (torch.nn.Module): Shared PyTorch model.
        optimizer (torch.optim.Optimizer): Shared optimizer.
        lock (mp.Lock): Lock for synchronizing access to the shared model.
        main_conn (mp.Connection): Connection to the main process.
        steps (int): Number of steps to run.
        video_log (bool): Whether to enable video logging.
        device (str): Device for computations ('cpu' or 'cuda').
        short_vid_dir (str): Directory to save short video files.
        max_frame_count (int): Number of frames to collect before saving a video.
    """
    try:
        print('Starting env_worker process...')
        # Initialize the environment
        env = HumanSurvival(**ENV_KWARGS).make()

        # Create an independent agent for the environment
        agent_parameters = pickle.load(open('./Model_Weights/2x_pre/2x.model', "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)

        # Move the shared model to the worker's device
        model.to(device)
        agent.model = model

        # Initialize environment and video frames
        obs = env.reset()
        frames = []
        chunk_count = 0

        for step in range(steps):
            # Forward pass to get the action
            with torch.no_grad():
                minerl_action = agent.get_action(obs)

            # Perform an environment step
            obs, reward, done, info = env.step(minerl_action)

            # Record video frame
            if video_log:
                frames.append(obs['pov'])

                # Save a short video chunk if the frame limit is reached
                if len(frames) == max_frame_count:
                    short_vid_name = os.path.join(short_vid_dir, f"worker_{os.getpid()}_video_{chunk_count}.mp4")
                    gen_vid(frames, short_vid_name)
                    chunk_count += 1
                    frames = []

            # Compute loss and backpropagate
            with lock:  # Lock the model for thread-safe updates
                optimizer.zero_grad()
                loss = torch.tensor(reward, dtype=torch.float32).to(device)  # Dummy loss; replace with real loss
                loss.backward()
                optimizer.step()

            # Send data to the main process
            main_conn.send((obs, reward, done, info))

            # Reset the environment if the episode ends
            if done:
                obs = env.reset()

    except Exception as e:
        print(f"env_worker encountered an error: {e}")
    finally:
        # Save remaining frames as a video
        if video_log and frames:
            short_vid_name = os.path.join(short_vid_dir, f"worker_{os.getpid()}_video_{chunk_count}.mp4")
            gen_vid(frames, short_vid_name)

        env.close()
        main_conn.close()


def gen_vid(frames, video_path, fps=15):
    """
    Generates a single video file from frames.
    Args:
        frames (list): List of 'pov' frames.
        video_path (str): Path to save the video file.
        fps (int): Frames per second for the output video.
    """
    if len(frames) > 0:
        frame_height, frame_width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"Video saved at: {video_path}")
    else:
        print(f"No frames available for video at {video_path}, skipping.")


def comp_vid(worker_id, input_path, output_path, fps=15):
    """
    Combines all short videos for a worker into a single video.
    Args:
        worker_id (int): ID of the worker.
        input_path (str): Directory containing short video chunks.
        output_path (str): Path to save the combined video.
        fps (int): Frames per second for the combined video.
    """
    video_files = sorted(
        f for f in os.listdir(input_path)
        if f.startswith(f"worker_{worker_id}_video") and f.endswith(".mp4"))

    if not video_files:
        print(f"No video files found for worker {worker_id}, skipping video compilation.")
        return

    frame_size = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for video_file in video_files:
        video_path = os.path.join(input_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if frame_size is None:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (frame_width, frame_height)
            out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        os.remove(video_path)  # Remove the short video after combining

    if out:
        out.release()
    print(f"Combined video for worker {worker_id} saved at: {output_path}")


def main(model_path, weights, num_envs, steps, video_log, max_frame_count, device):
    """
    Main function to run multiple environments in parallel with a shared model and video logging.
    """
    mp.set_start_method('spawn', force=True)  # Required for torch.multiprocessing

    # Load the shared model
    # shared_model = MineRLAgent.load_model(model)
    shared_model = pickle.load(open(model_path, "rb"))
    shared_model.share_memory_()  # Share model across processes

    # Create a shared optimizer
    shared_optimizer = torch.optim.Adam(shared_model.parameters(), lr=0.001)

    # Create a lock for synchronization
    lock = mp.Lock()

    # Create logging directories
    current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
    session_path = os.path.join("./Model_weights/pre_log", current_time)
    os.makedirs(session_path, exist_ok=True)
    save_path = os.path.join(session_path, "obs_rew")
    os.makedirs(save_path, exist_ok=True)

    # Video logging directories
    video_dir, short_vid_dir = None, None
    if video_log:
        video_dir = os.path.join(session_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        short_vid_dir = os.path.join(session_path, "short_videos")
        os.makedirs(short_vid_dir, exist_ok=True)

    # Launch worker processes
    processes, parent_conns = [], []
    for i in range(num_envs):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(
            target=env_worker,
            args=(shared_model, shared_optimizer, lock, child_conn, steps, video_log, device, short_vid_dir, max_frame_count)
        )
        process.start()
        parent_conns.append(parent_conn)
        processes.append(process)

    obs_inv, rewards = [], []
    try:
        while True:
            for i, conn in enumerate(parent_conns):
                if conn.poll():
                    obs, reward, done, info = conn.recv()
                    obs_inv.append(obs['inventory'])
                    rewards.append(reward)

            if all(not conn.poll() for conn in parent_conns):
                print("All workers are idle. Exiting...")
                break

    except KeyboardInterrupt:
        print("Shutting down environments...")
    finally:
        # Save rewards and observations
        np.save(os.path.join(save_path, "reward.npy"), np.array(rewards))
        np.save(os.path.join(save_path, "obs_inv.npy"), np.array(obs_inv))

        # Combine videos for each worker
        if video_log:
            for i, process in enumerate(processes):
                final_vid_path = os.path.join(video_dir, f"worker_{process.pid}_final_vid.mp4")
                comp_vid(process.pid, short_vid_dir, final_vid_path)
            shutil.rmtree(short_vid_dir, ignore_errors=True)

        for conn in parent_conns:
            conn.close()
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    args = arg_parser()
    device = "cuda" if args.cuda else "cpu"
    main(args.model, args.weights, args.num_envs, args.steps, args.videolog, args.max_frame_count, device)
