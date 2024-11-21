import os
import pickle
import shutil
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.env_specs.mlg import MLGWaterEnvSpec
from agent import MineRLAgent, ENV_KWARGS
import cv2


def arg_parser():
    """
    Parse command-line arguments for the MineRL environment runner.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser("Run pretrained models on MineRL environment in parallel without stable-baselines3")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments to run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run in each environment.")
    parser.add_argument("--videolog", action='store_true', help="Enable video logging")
    parser.add_argument("--weights", type=str, default="./Model_Weights/2x_pre/rl-from-house-2x.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="./Model_Weights/2x_pre/2x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--max_frame_count", type=int, default=100, help="Number of frames to collect before saving a video.")
    return parser.parse_args()


def env_worker(model_path, weights_path, main_conn, steps, video_log, video_path=None):
    """
    Worker function to run a single instance of the MineRL environment.
    Args:
        model_path (str): Path to the model file.
        weights_path (str): Path to the weights file.
        main_conn (multiprocessing.Connection): Connection to the main process.
        steps (int): Number of steps to run.
        video_log (bool): Whether to enable video logging.
        video_path (str): Path to save videos (if applicable).
    """

    print('Starting env_worker process...')
    try:
       
        # Initialize the environment
        env = HumanSurvival(**ENV_KWARGS).make()

        # # MLG Custom Environment
        # abs_MLG = MLGWaterEnvSpec()
        # abs_MLG.register()
        # env = gym.make('MLGWater-v0')

        # Load agent and model
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
        agent.load_weights(weights_path)

        obs = env.reset()

        for step in range(steps):
            # Get the action for the current observation
            minerl_action = agent.get_action(obs)
            obs, reward, done, info = env.step(minerl_action)

            # Send data to the main process (e.g., for rewards tracking)
            main_conn.send((obs, reward, done, info))

            # Reset the environment if the episode ends
            if done:
                obs = env.reset()

    except Exception as e:
        print(f"env_worker encountered an error: {e}")
    finally:
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
        if f.startswith(f"env_{worker_id}_video") and f.endswith(".mp4"))

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


def main(model, weights, num_envs, steps, video_log, max_frame_count):
    """
    Main function to run multiple environments in parallel.
    Args:
        model (str): Path to the model file.
        weights (str): Path to the weights file.
        num_envs (int): Number of environments to run in parallel.
        steps (int): Number of steps to run in each environment.
        video_log (bool): Whether to enable video logging.
        max_frame_count (int): Maximum number of frames per video chunk.
    """
    mp.set_start_method('spawn', force=True)  # Required for torch.multiprocessing

    # Create a session for logging
    current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
    session_path = os.path.join("./Model_weights/pre_log", current_time)
    os.makedirs(session_path, exist_ok=True)
    print(f"Logging session created at: {session_path}")

    save_path = os.path.join(session_path, "obs_rew")
    os.makedirs(save_path, exist_ok=True)

    if video_log:
        video_dir = os.path.join(session_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        short_vid_dir = os.path.join(session_path, "short_videos")
        os.makedirs(short_vid_dir, exist_ok=True)

    processes, parent_conns = [], []
    frames = {i: [] for i in range(num_envs)}
    chunk_count = {i: 0 for i in range(num_envs)}
    worker_steps = {i: 0 for i in range(num_envs)}

    # Start each environment in its own process
    for i in range(num_envs):
        print(f"Starting env_worker for env {i}")
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=env_worker, args=(model, weights, child_conn, steps, video_log))
        process.start()
        parent_conns.append(parent_conn)
        processes.append(process)

    obs_inv, rewards = [], []

    try:
        while any(worker_steps[i] < steps for i in range(num_envs)):
            for i, conn in enumerate(parent_conns):
                if worker_steps[i] >= steps:
                    continue

                if conn.poll():
                    obs, reward, done, info = conn.recv()
                    obs_inv.append(obs['inventory'])
                    rewards.append(reward)
                    frames[i].append(obs['pov'])
                    worker_steps[i] += 1

                    if len(frames[i]) == max_frame_count:
                        short_vid_name = os.path.join(short_vid_dir, f"env_{i}_video_{chunk_count[i]}.mp4")
                        gen_vid(frames[i], short_vid_name)
                        chunk_count[i] += 1
                        frames[i] = []

            if all(worker_steps[i] >= steps for i in range(num_envs)):
                print(f"All workers have completed {steps} steps.")
                break

    except KeyboardInterrupt:
        print("Shutting down environments...")
    finally:
        # Save observations and rewards to files
        np.save(os.path.join(save_path, "reward.npy"), np.array(rewards))
        np.save(os.path.join(save_path, "obs_inv.npy"), np.array(obs_inv))
        print('cum_reward:', sum(rewards))

        for conn in parent_conns:
            conn.close()
        for process in processes:
            process.terminate()
            process.join()

        if video_log:
            for i in range(num_envs):
                final_vid_path = os.path.join(video_dir, f"env_{i}_final_vid.mp4")
                comp_vid(i, short_vid_dir, final_vid_path)
            shutil.rmtree(short_vid_dir, ignore_errors=True)


if __name__ == "__main__":
    args = arg_parser()
    main(args.model, args.weights, args.num_envs, args.steps, args.videolog, args.max_frame_count)
