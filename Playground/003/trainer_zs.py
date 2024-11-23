import argparse
import datetime
import os
# os.add_dll_directory("C://Users//yhttmb//.mujoco//mujoco210//bin")
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from PPOagent import Agent
from buffer import Buffer
from utils import *

def load_checkpoint(agent, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    agent.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from checkpoint: {checkpoint_path} at epoch {start_epoch}")
    return start_epoch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--env", default="Humanoid-v4", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=48, help="Number of environments") # Parallel
    parser.add_argument("--n-epochs", type=int, default=3000, help="Number of epochs to run") # 3000 is fully trained
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=1e-5, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.999, help="Multiply with lr every epoch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--reward-scale", type=float, default=0.005, help="Reward scaling")
    parser.add_argument("--render-epoch", type=int, default=200, help="Render every n-th epoch")
    parser.add_argument("--save-epoch", type=int, default=200, help="Save the model every n-th epoch")
    # NOTE: add checkpoint path as argument
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to resume checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    args.device = device

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create the environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    args.obs_dim = obs_dim
    args.act_dim = act_dim

    agent = Agent(args).to(device)
    optimizer = agent.optimizer
    args.agent = agent

    # NOTE: Resume from checkpoint if specified
    start_epoch = 1
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_epoch = load_checkpoint(agent, optimizer, args.checkpoint_path)

    print("Actor network (mu) architecture:\n", agent.actor)
    print("Critic network architecture:\n", agent.critic)

    # Start the training
    global_step_idx = 0
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.tensor([float(False)] * args.n_envs, device=device)
    next_truncateds = torch.tensor([float(False)] * args.n_envs, device=device)

    reward_list = []

    try:
        for epoch in range(1, args.n_epochs + 1):
            # Collect trajectories
            for step_idx in range(0, args.n_steps):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds
                agent.store_to_buffer(obs, envs, reward_list, terminateds, truncateds)

            traj_obs, traj_act, traj_logprob, traj_adv, traj_ret, traj_val, traj_indices = agent.sample_batch(next_obs, next_terminateds, next_truncateds)
            sum_loss_total, sum_loss_policy, sum_loss_value, sum_entropy = agent.update(traj_obs, traj_act, traj_logprob, traj_adv, traj_ret, traj_indices)

            # Log info on console
            avg_reward = sum(reward_list) / len(reward_list)
            # Rescale the rewards
            avg_reward /= args.reward_scale
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s. "
                  f"Avg reward: {avg_reward:.2f}. ")
            reward_list = []

            # Every n epochs, log the video
            if epoch % args.render_epoch == 0 or epoch == 1:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Every n epochs, save the model
            if epoch % args.save_epoch == 0 or epoch == 1:
                # torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_{epoch}.dat"))
                # NOTE: save model with epoch number
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.dat")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

            # Log everything to tensorboard
            writer.add_scalar("losses/policy_loss", sum_loss_policy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/value_loss", sum_loss_value / args.train_iters, global_step_idx)
            writer.add_scalar("losses/entropy", sum_entropy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/total_loss", sum_loss_total / args.train_iters, global_step_idx)
            writer.add_scalar("charts/avg_reward", avg_reward, global_step_idx)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step_idx)
            writer.add_scalar("charts/SPS", global_step_idx / (time.time() - start_time), global_step_idx)

    finally:
        # Close the environments and tensorboard writer
        envs.close()
        test_env.close()
        writer.close()

        # Save the model
        torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "model.dat"))
