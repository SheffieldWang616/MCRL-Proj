import os
import gym
from multiprocessing import Process, Pipe
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from agent import Agent

os.add_dll_directory("C:\\Users\\Sheffield\\.mujoco\\mujoco210\\bin")


def arg_parser():
    parser = ArgumentParser("Run PPO training on Humanoid environment in parallel")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments to run.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run in each environment.")
    parser.add_argument("--max_frame_count", type=int, default=100, help="Number of frames to collect before saving a video.")
    return parser.parse_args()


def env_worker(worker_id, conn, steps):
    """
    Worker function to handle environment interactions in parallel.
    """
    try:
        # Create the Humanoid environment
        env = gym.make("Humanoid-v2")
        
        # Define PPO parameters (these should match the ones required in the agent.py implementation)
        params = {
            'n_layers': 2,               # Number of layers in the MLP
            'size': 64,                  # Hidden layer size
            'learning_rate': 3e-4,       # Learning rate
            'learning_rate_decay': 0.99, # Learning rate decay (if applicable)
            'n_steps': 2048,             # Number of steps per update
            'n_envs': 1,                 # Number of parallel environments (1 for this worker)
            'device': 'cpu',             # Change to 'cuda' if using GPU
            'gamma': 0.99,               # Discount factor
            'gae_lambda': 0.95,          # GAE lambda
            'actor_update_steps': 4,     # Number of policy updates
            'critic_update_steps': 4,    # Number of value updates
            'clip_eps': 0.2              # Clipping parameter for PPO
        }

        # Initialize the PPO Agent
        agent = Agent(env=env, params=params)
        
        # Reset the environment
        obs = env.reset()
        trajectory = []

        # Interact with the environment for the specified number of steps
        for step in range(steps):
            action = agent.select_action(obs)  # Get action from PPO policy
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, next_obs, done))
            obs = next_obs

            # Send rewards and done status to the main process
            conn.send((reward, done))
            if done:
                obs = env.reset()

        # Send trajectory data to the main process
        conn.send(("trajectory", trajectory))

    except Exception as e:
        print(f"Worker {worker_id} encountered an error: {e}")
    finally:
        conn.close()
        env.close()


def main(num_envs, steps, max_frame_count):
    """
    Main function to run PPO training in parallel environments.
    """
    processes, parent_conns = [], []
    trajectories = []

    for i in range(num_envs):
        parent_conn, child_conn = Pipe()
        process = Process(target=env_worker, args=(i, child_conn, steps))
        process.start()
        parent_conns.append(parent_conn)
        processes.append(process)

    # Collect data from workers
    try:
        for i, conn in enumerate(parent_conns):
            while True:
                data = conn.recv()
                if isinstance(data, tuple) and data[0] == "trajectory":
                    trajectories.append(data[1])
                    break  # Exit when trajectory is received
                else:
                    reward, done = data
                    # Handle rewards, done signals as needed

    except KeyboardInterrupt:
        print("Training interrupted. Shutting down workers...")

    finally:
        # Close all processes and connections
        for conn in parent_conns:
            conn.close()
        for process in processes:
            process.terminate()
            process.join()

    # Update PPO policy with collected trajectories
    if trajectories:
        agent = Agent(None)  # Replace with actual PPO agent
        for trajectory in trajectories:
            agent.update_policy(trajectory)  # Update policy using collected data

    print("Training complete.")


if __name__ == "__main__":
    args = arg_parser()
    main(args.num_envs, args.steps, args.max_frame_count)
