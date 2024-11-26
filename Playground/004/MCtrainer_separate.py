import os
import argparse
import datetime
import time
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle
from MCagent import MineRLAgent
from buffer import Buffer
from utils import *

# Function 1: Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default="Tree-v0", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=3000, help="Number of epochs")
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=1e-5, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.999, help="Learning rate decay per epoch")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    return parser.parse_args()

# Function 2: Initialize Agent
def initialize_agent(model_path, weights_path, device):
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    return agent

# Function 3: Setup Environments and Replay Buffer
def setup_envs_and_buffer(env_name, n_envs, n_steps, agent, gamma, gae_lambda, device):
    envs = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
    replay_buffer = Buffer(envs.single_observation_space.shape, n_steps, n_envs, device, gamma, gae_lambda, agent)
    return envs, replay_buffer

# Function 4: Training Loop
def train_loop(args, agent, envs, replay_buffer, writer):
    optimizer = optim.Adam(agent.policy.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    obs = envs.reset()
    reward_list = []

    for epoch in tqdm(range(1, args.n_epochs + 1), desc="Epochs"):
        for step in tqdm(range(args.n_steps), desc="Collecting Trajectory", leave=False):
            actions = agent.get_action(obs)
            next_obs, rewards, dones, infos = envs.step(actions)
            replay_buffer.store(obs, actions, rewards, dones)
            obs = next_obs

        with torch.no_grad():
            traj_adv, traj_ret = replay_buffer.calculate_advantages()
            traj_obs, traj_act, traj_val, traj_logprob = replay_buffer.get()

        train_batches(
            agent, optimizer, traj_obs, traj_act, traj_adv, traj_ret, traj_logprob, args.batch_size, args.train_iters
        )
        scheduler.step()

        writer.add_scalar("charts/reward", sum(reward_list), epoch)
        reward_list = []

    envs.close()

# Function 5: Train Batches
def train_batches(agent, optimizer, traj_obs, traj_act, traj_adv, traj_ret, traj_logprob, batch_size, train_iters):
    traj_indices = np.arange(len(traj_obs))
    for _ in range(train_iters):
        np.random.shuffle(traj_indices)
        for start in range(0, len(traj_obs), batch_size):
            end = start + batch_size
            batch_indices = traj_indices[start:end]
            batch_obs, batch_act, batch_adv = traj_obs[batch_indices], traj_act[batch_indices], traj_adv[batch_indices]
            _, new_logprobs, new_values, _, entropies = agent.get_logprob_and_value(batch_obs, batch_act)

            ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])
            policy_loss = -torch.min(ratios * batch_adv, torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * batch_adv).mean()
            value_loss = ((new_values - traj_ret[batch_indices]) ** 2).mean()
            entropy_loss = -entropies.mean()

            loss = policy_loss + value_loss - entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Main Function
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Logging and directories
    current_dir = os.path.dirname(__file__)
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    writer = SummaryWriter(log_dir=os.path.join(current_dir, "logs", run_name))

    # Initialize agent, environments, and buffer
    agent = initialize_agent(args.model, args.weights, device)
    envs, replay_buffer = setup_envs_and_buffer(args.env, args.n_envs, args.n_steps, agent, args.gamma, args.gae_lambda, device)

    # Train
    train_loop(args, agent, envs, replay_buffer, writer)
