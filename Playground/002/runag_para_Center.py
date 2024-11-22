import os
import pickle
import shutil
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from multiprocessing import Process, Pipe, Manager

import cv2
import gym
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS


def arg_parser():
    """
    Parses command-line arguments for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
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


def env_worker(env_id, main_conn, steps):
    """
    Worker process to interact with the environment and send observations/results to the main process.

    Args:
        env_id (int): ID of the worker environment.
        main_conn (Pipe): Connection to communicate with the main process.
        steps (int): Number of steps to run in the environment.
    """
    print(f"Starting env_worker {env_id}")
    try:
        # Initialize the environment
        env = HumanSurvival(**ENV_KWARGS).make()
        obs = env.reset()  # Reset the environment and get the initial observation

        for step in range(steps):
            # Send the current observation to the main process
            main_conn.send(("obs", obs))

            # Wait for the action from the main process
            action = main_conn.recv()

            # Perform the environment step
            obs, reward, done, info = env.step(action)

            # Send the step results to the main process
            main_conn.send(('result',(obs, reward, done, info)))

            if done:
                obs = env.reset()  # Reset the environment if the episode ends

    except Exception as e:
        print(f"Worker {env_id} encountered an error: {e}")
    finally:
        env.close()
        main_conn.close()

def gen_vid(frames, video_path, fps=15):
    """
    Generates a single video file from a list of frames.

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
        f for f in os.listdir(input_path) if f.startswith(f"env_{worker_id}_video") and f.endswith(".mp4")
    )

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



def main(model_path, weights_path, num_envs, steps, video_log, max_frame_count):
    """
    Main function to manage multiple environments with a shared model.

    Args:
        model_path (str): Path to the saved model.
        weights_path (str): Path to the saved model weights.
        num_envs (int): Number of parallel environments to run.
        steps (int): Number of steps to run in each environment.
        video_log (bool): Flag to enable/disable video logging.
        max_frame_count (int): Number of frames to collect before saving a video chunk.
    """
    # Load the shared model
    with open(model_path, "rb") as model_file:
        agent_parameters = pickle.load(model_file)
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(None, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs,device='cuda' if args.cuda else 'cpu')
    agent.load_weights(weights_path)

    # Directory setup
    current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
    session_path = os.path.join(".\\Model_weights\\pre_log", current_time)
    os.makedirs(session_path, exist_ok=True)

    save_path = os.path.join(session_path, "obs_rew")
    os.makedirs(save_path, exist_ok=True)

    if video_log:
        video_dir = os.path.join(session_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        short_vid_dir = os.path.join(session_path, "short_videos")
        os.makedirs(short_vid_dir, exist_ok=True)

    # Start worker processes
    processes, parent_conns = [], []
    frames = {i: [] for i in range(num_envs)}  # To store frames for each worker
    chunk_count = {i: 0 for i in range(num_envs)}  # Counter for video chunks
    worker_steps = {i: 0 for i in range(num_envs)}  # Steps counter for each worker
    obs_inv, rewards = [], []  # To store observations and rewards across workers

    for i in range(num_envs):
        parent_conn, child_conn = Pipe()
        process = Process(target=env_worker, args=(i, child_conn, steps))
        process.start()
        parent_conns.append(parent_conn)
        processes.append(process)

    # Main loop to collect data
    try:
        while any(worker_steps[i] < steps for i in range(num_envs)):
            for i, conn in enumerate(parent_conns):
                if worker_steps[i] >= steps:
                    continue  # Skip workers that have completed their steps

                if conn.poll():  # Check if there's data to receive
                    msg_type, data = conn.recv()

                    if msg_type == "obs":
                        action = agent.get_action(data)
                        conn.send(action)
                    elif msg_type == "result":
                        obs,reward,done,info = data
                        obs_inv.append(obs['inventory'])
                        rewards.append(reward)
                        frames[i].append(obs['pov'])
                        worker_steps[i] += 1 # increment the worker steps

                    if len(frames[i]) == max_frame_count:
                        # gen vid for a short video and save it to short_vid_dir
                        short_vid_name = os.path.join(short_vid_dir, f"env_{i}_video_{chunk_count[i]}.mp4") # short video name
                        gen_vid(frames[i], short_vid_name) # generate short video
                        chunk_count[i] += 1
                        frames[i] = [] # clear the frames list

                    # Exit the loop after 100 steps
            if all(worker_steps[i] >= steps for i in range(num_envs)):
                print(f"All workers have completed {steps} steps.")
                break

    except KeyboardInterrupt:
        print("Shutting down environments...")
    finally:
        # Save observations and rewards to files
        print(f"Saving {len(obs_inv)} observations and {len(rewards)} rewards for {num_envs} different Paralleled Workers.")
        np.save(os.path.join(save_path, "reward.npy"), np.array(rewards))
        np.save(os.path.join(save_path, "obs_inv.npy"), np.array(obs_inv))
        
        print('cum_reward:', sum(rewards))
        for conn in parent_conns:
            conn.close()  # Close all connections after main loop ends
        for process in processes:
            process.terminate()
            process.join()

        if video_log:
            
            print(f"Saving videos {i} to: {video_dir}")
            
            # Generate final videos from the short videos collected in each environment, saved in video_dir
            for i in range(num_envs):
                final_vid_path = os.path.join(video_dir, f"env_{i}_final_vid.mp4")
                comp_vid(i, short_vid_dir, final_vid_path)
            
            # remove short videos
            shutil.rmtree(short_vid_dir, ignore_errors=True)


if __name__ == "__main__":
    args = arg_parser()
    main(args.model, args.weights, args.num_envs, args.steps, args.videolog, args.max_frame_count)
