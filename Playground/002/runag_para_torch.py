import os
import pickle
import shutil
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS
import cv2
from MC_Torch_model import ReconstructedModel


def arg_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser("Run pretrained models on MineRL environment in parallel with torch.multiprocessing")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments to run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run in each environment.")
    parser.add_argument("--videolog", action='store_true', help="Enable video logging")
    parser.add_argument("--weights", type=str, default="./Model_Weights/2x_pre/rl-from-house-2x.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="./Model_Weights/2x_pre/2x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--max_frame_count", type=int, default=100, help="Number of frames to collect before saving a video.")
    return parser.parse_args()


def env_worker(shared_model, lock, main_conn, steps, video_log, max_frame_count, short_vid_dir=None):
    """
    Worker function to run a single instance of the MineRL environment.
    """
    print(f"Worker {os.getpid()} started...")
    try:
        # Initialize the environment
        env = HumanSurvival(**ENV_KWARGS).make()

        # Initialize agent with shared model
        agent = MineRLAgent(None, policy_kwargs={})
        agent.model = shared_model  # Use the shared model passed from main

        obs = env.reset()
        frames = []
        chunk_count = 0

        for step in range(steps):
            with lock:  # Lock access to shared model
                minerl_action = agent.get_action(obs)

            # Take a step in the environment
            obs, reward, done, info = env.step(minerl_action)

            # Send data back to the main process
            main_conn.send((obs, reward, done, info))

            # Record video if logging is enabled
            if video_log:
                frames.append(obs["pov"])
                if len(frames) == max_frame_count:
                    short_vid_name = os.path.join(short_vid_dir, f"worker_{os.getpid()}_video_{chunk_count}.mp4")
                    gen_vid(frames, short_vid_name)
                    chunk_count += 1
                    frames = []

            if done:
                obs = env.reset()

    except Exception as e:
        print(f"Worker {os.getpid()} encountered an error: {e}")
    finally:
        if video_log and frames:
            short_vid_name = os.path.join(short_vid_dir, f"worker_{os.getpid()}_video_{chunk_count}.mp4")
            gen_vid(frames, short_vid_name)
        env.close()
        main_conn.close()


def gen_vid(frames, video_path, fps=15):
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


def main(model_path, weights_path, num_envs, steps, video_log, max_frame_count):
    """
    Main function to manage multiprocessing for the MineRL environment.
    """
    mp.set_start_method('spawn', force=True)

    # Load model arguments
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    policy_kwargs = model_data["model"]["args"]["net"]["args"]
    pi_head_kwargs = model_data["model"]["args"]["pi_head_opts"]

    # Create shared model
    # agent = MineRLAgent(None, policy_kwargs=policy_kwargs)
    # agent.model = ReconstructedModel(policy_kwargs)
    # agent.load_weights(weights_path)
    # shared_model = agent.model
    # shared_model.share_memory_()  # Make the model multiprocessing-compatible
    shared_model = ReconstructedModel(policy_kwargs)
    shared_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    shared_model.share_memory_()  # Make the model multiprocessing-compatible


    # Create a lock for synchronizing access
    lock = mp.Lock()

    # Logging setup
    current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
    session_path = os.path.join("./Model_weights/pre_log", current_time)
    os.makedirs(session_path, exist_ok=True)

    short_vid_dir = None
    if video_log:
        short_vid_dir = os.path.join(session_path, "short_videos")
        os.makedirs(short_vid_dir, exist_ok=True)

    # Start worker processes
    processes, parent_conns = [], []
    for i in range(num_envs):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(
            target=env_worker,
            args=(shared_model, lock, child_conn, steps, video_log, max_frame_count, short_vid_dir)
        )
        process.start()
        parent_conns.append(parent_conn)
        processes.append(process)

    try:
        while any(process.is_alive() for process in processes):
            for conn in parent_conns:
                if conn.poll():
                    obs, reward, done, info = conn.recv()
                    print(f"Reward received: {reward}")

    except KeyboardInterrupt:
        print("Shutting down environments...")
    finally:
        for conn in parent_conns:
            conn.close()
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    args = arg_parser()
    main(args.model, args.weights, args.num_envs, args.steps, args.videolog, args.max_frame_count)
