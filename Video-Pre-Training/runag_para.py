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
from minerl.herobraine.env_specs.mlg import MLGWaterEnvSpec
from agent import MineRLAgent, ENV_KWARGS

def arg_parser():
    parser = ArgumentParser("Run pretrained models on MineRL environment in parallel without stable-baselines3")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments to run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run in each environment.")
    parser.add_argument("--videolog", action='store_true', help="Enable video logging")
    parser.add_argument("--weights", type=str, default="./Model_Weights/2x_pre/rl-from-house-2x.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="./Model_Weights/2x_pre/2x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--max_frame_count", type=int, default=100, help="Number of frames to collect before saving a video.")
    
    return parser.parse_args()

def env_worker(model_path, weights_path, main_conn, steps, video_log, video_path = None):
    print('starting env_worker process')
    try:
        # MineRL Hunam Survival Environment
        env = HumanSurvival(**ENV_KWARGS).make()

        # # MLG Custom Environment
        # abs_MLG = MLGWaterEnvSpec()
        # abs_MLG.register()
        # env = gym.make('MLGWater-v0')

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
        # print(f"Video saved at: {video_path} with {len(frames)} frames")
    else:
        print(f"No frames available for video at {video_path}, skipping.")


def comp_vid(worker_id, input_path, output_path, fps = 15):
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
    # print(f"Combined video saved at: {output_path}")
    print(f"Combined video for worker {worker_id} saved at: {output_path}")


def main(model, weights, num_envs, steps, video_log, max_frame_count):
    # Create a list to store process connections and processes

    # print("Starting main function")
    current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
    session_path = os.path.join(".\\Model_weights\\pre_log", current_time)
    os.makedirs(session_path, exist_ok=True)
    print(f"Logging session created at: {session_path}")

    save_path = os.path.join(session_path, "obs_rew")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving logs to: {save_path}")

    # Path where final videos will be saved
    if video_log:
        # Final video directory
        video_dir = os.path.join(session_path, "videos")
        os.makedirs(video_dir, exist_ok=True)
        # Short video directory
        short_vid_dir = os.path.join(session_path, "short_videos")
        os.makedirs(short_vid_dir, exist_ok=True)

    processes, parent_conns = [], []
    frames = {i: [] for i in range(num_envs)}
    chunk_count = {i: 0 for i in range(num_envs)}
    worker_steps = {i: 0 for i in range(num_envs)}

    # Start each environment in its own process
    for i in range(num_envs):
        print(f"Starting env_worker for env {i}")
        parent_conn, child_conn = Pipe()
        # env_worker(model_path, weights_path, main_conn, steps, video_log, video_path = None)
        process = Process(target=env_worker, args=(model, weights, child_conn,steps, None))
        process.start()
        # Keep track of parent connections and processes
        parent_conns.append(parent_conn)
        processes.append(process)

    # initialize variables for saving observations and rewards
    obs_inv, rewards=[], []

    # Collect data from each environment in parallel
    try:
        while any(worker_steps[i] < steps for i in range(num_envs)):
            for i, conn in enumerate(parent_conns):
                if worker_steps[i] >= steps:
                    continue  # Skip workers that have completed their steps

                if conn.poll():  # Check if there's data to receive
                    obs, reward, done, info = conn.recv()
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
        print(f"Saving {len(obs)} observations and {len(rewards)} rewards for {num_envs} different Paralleled Workers.")
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


# -----------------------------------------gen_vid for env using frames-----------------------------------------

# def gen_vid(frames, video_dir, fps=15):
#     """
#     Generates video files from the 'pov' frames collected in each environment.
    
#     Args:
#         frames (dict): Dictionary where each key is an environment ID (e.g., 'env_0') and each value is a list of 'pov' frames.
#         video_dir (str): Path to the directory where video files will be saved.
#         fps (int): Frames per second for the output video.
#     """
#     for env_id, env_frames in frames.items():
#         # Define the video path for this environment
#         video_path = os.path.join(video_dir, f"{env_id}_video.mp4")
        
#         # Check that there are frames available for this environment
#         if len(env_frames) > 0:
#             # Get frame size from the first frame
#             frame_height, frame_width, _ = env_frames[0].shape
            
#             # Initialize the VideoWriter with the codec, FPS, and frame size
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
#             # Write each frame to the video
#             for frame in env_frames:
#                 # Convert RGB to BGR for OpenCV
#                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 video_writer.write(frame_bgr)
            
#             # Release the video writer for this environment
#             video_writer.release()
#             print(f"Video saved for worker {env_id} at {video_path}")
#         else:
#             print(f"No frames available for {env_id}, skipping video creation.")



# ----------------------------------------- main function -----------------------------------------

# def main(model, weights, num_envs, steps, video_log, max_frame_count):
#     # Create directories for logs and videos
#     current_time = datetime.now().strftime("session_%m_%d_%y_%H-%M")
#     session_path = os.path.join(".\\Model_weights\\pre_log", current_time)
#     os.makedirs(session_path, exist_ok=True)
#     print(f"Logging session created at: {session_path}")

#     save_path = os.path.join(session_path, "obs_rew")
#     os.makedirs(save_path, exist_ok=True)
#     print(f"Saving logs to: {save_path}")

#     if video_log:
#         video_dir = os.path.join(session_path, "videos")
#         os.makedirs(video_dir, exist_ok=True)
#         short_vid_dir = os.path.join(session_path, "short_videos")
#         os.makedirs(short_vid_dir, exist_ok=True)

#     processes, parent_conns = [], []
#     frames = {i: [] for i in range(num_envs)}
#     chunk_count = {i: 0 for i in range(num_envs)}
#     worker_steps = {i: 0 for i in range(num_envs)}

#     # Start each environment in its own process
#     for i in range(num_envs):
#         print(f"Starting env_worker for env {i}")
#         parent_conn, child_conn = Pipe()
#         process = Process(target=env_worker, args=(model, weights, child_conn, steps, video_log))
#         process.start()
#         parent_conns.append(parent_conn)
#         processes.append(process)

#     try:
#         while any(worker_steps[i] < steps for i in range(num_envs)):  # Run until all workers finish
#             for i, conn in enumerate(parent_conns):
#                 if worker_steps[i] >= steps:
#                     continue  # Skip workers that have completed their steps

#                 if conn.poll():  # Check for new data
#                     obs, reward, done, info = conn.recv()
#                     worker_steps[i] += 1
#                     frames[i].append(obs['pov'])

#                     # Generate short video if frame limit is reached
#                     if len(frames[i]) == max_frame_count:
#                         short_vid_name = os.path.join(short_vid_dir, f"env_{i}_video_{chunk_count[i]}.mp4")
#                         gen_vid(frames[i], short_vid_name)
#                         chunk_count[i] += 1
#                         frames[i] = []  # Clear the frames list

#                     if done:
#                         obs = env.reset()

#     except KeyboardInterrupt:
#         print("Shutting down environments...")
#     finally:
#         for conn in parent_conns:
#             conn.close()
#         for process in processes:
#             process.terminate()
#             process.join()

#         if video_log:
#             print(f"Saving videos to: {video_dir}")
#             for i in range(num_envs):
#                 final_vid_path = os.path.join(video_dir, f"env_{i}_final_vid.mp4")
#                 comp_vid(i, short_vid_dir, final_vid_path)
#             shutil.rmtree(short_vid_dir, ignore_errors=True)


# ------------------------------------------ env_worker function ------------------------------------------

# def env_worker(model_path, weights_path, main_conn, steps, video_log, video_path = None):
#     print('starting env_worker process')
#     try:
#         # env = HumanSurvival(**ENV_KWARGS).make()
#         abs_MLG = MLGWaterEnvSpec()
#         abs_MLG.register()
#         env = gym.make('MLGWater-v0')
#         agent_parameters = pickle.load(open(model_path, "rb"))
#         policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
#         pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
#         pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
#         agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
#         agent.load_weights(weights_path)

#         obs = env.reset()

#         while True:
#             # print('getting obs')
#             minerl_action = agent.get_action(obs)
#             obs, reward, done, info = env.step(minerl_action)

#             # Send data to the main process (e.g., for rewards tracking)
#             # print('sending data to main_conn')
#             main_conn.send((obs, reward, done, info))
            
#             # Send only the 'pov' frame data to the video logger process (Commented Out)
#             # if video_conn:
#             #     print('sending pov to video_conn')
#             #     video_conn.send(obs['pov'])  # Send only the 'pov' frame for video logging

#             if done:
#                 # print('episode done, resetting env')
#                 obs = env.reset()
#                 # if video_conn:
#                 #     video_conn.send(None)  # Signal the end of an episode for video logging
#     except Exception as e:
#         print(f"env_worker encountered an error: {e}")
#     finally:
#         # print('closing env')
#         env.close()
#         main_conn.close()
#         # if video_conn:
#         #     video_conn.close()