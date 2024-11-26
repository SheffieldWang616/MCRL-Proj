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
    parser.add_argument("--run-name", default="my_run", help="Name of the run")
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default="Tree-v0", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments") # default 1
    parser.add_argument("--n-epochs", type=int, default=1, help="Number of epochs") # default 3000
    parser.add_argument("--n-steps", type=int, default=50, help="Number of steps per epoch per environment") # default 1024
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=1e-5, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.999, help="Learning rate decay per epoch")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--weights", type=str,default ="./Model_Weights/2x_pre/2x.weights", help="Path to weights file")
    parser.add_argument("--model", type=str, default="./Model_Weights/2x_pre/2x.model", help="Path to model file")
    return parser.parse_args()

# Initialize Agent, Optimizer, Scheduler, Environments, and Test Environment
def initialize(args, device):
    # setting up agent
    agent_parameters = pickle.load(open(args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.weights)
    print("Agent initialized.")
    
    # setting up optimizer and scheduler
    optimizer = optim.Adam(agent.policy.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    print("Optimizer and scheduler initialized.")

    # setting up train and test environments
    envs = gym.make(args.env)
    test_env = gym.make(args.env)
    envs.seed(2143)
    test_env.seed(2143)
    print("Environments initialized.")

    return agent, optimizer, scheduler, envs, test_env


# Train Batches
def train_batches(traj_obs, traj_act, traj_adv, traj_ret, traj_logprob, agent, optimizer, scheduler, args):
    traj_indices = np.arange(args.n_steps * args.n_envs)
            
    sum_loss_policy = 0.0
    sum_loss_value = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    for _ in range(args.train_iters):

        np.random.shuffle(traj_indices)

        for start_idx in range(0, len(traj_obs), args.batch_size):
            end_idx = start_idx+ args.batch_size
            batch_indices = traj_indices[start_idx:end_idx].tolist()

            _, new_logprobs, new_values, _, entropies = agent.get_logprob_and_value(traj_obs[batch_indices], traj_act[batch_indices])

            ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])
            batch_adv = traj_adv[batch_indices]
            batch_adv = (batch_adv - batch_adv.mean()) / torch.max(batch_adv.std(), torch.tensor(1e-5, device=device))
            
            policy_loss1 = -batch_adv * ratios
            policy_loss2 = -batch_adv * torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()

            value_loss = 0.5 * ((new_values - traj_ret[batch_indices]) ** 2).mean()

            entropy = entropies.mean()
            
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(agent.policy.parameters(), args.max_grad_norm)
            optimizer.step()

            sum_loss_policy += policy_loss.item()
            sum_loss_value += value_loss.item()
            sum_entropy += entropy.item()
            sum_loss_total += loss.item()
        # break
    # break
    scheduler.step()

# Training Loop
def training_loop(args, agent, envs, device, writer):

    
    init_obs = envs.reset()
    next_obs = init_obs['pov']
    obs_dim = next_obs.shape
    replay_buffer = Buffer(obs_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, agent=agent)   
    next_obs = torch.tensor(np.array(next_obs, dtype=np.uint8), device=device)
    next_terminateds = torch.tensor([float(False)], device=device)

    global_step_idx = 0
    reward_list = []

    try:

        for epoch in tqdm(range(1, args.n_epochs + 1), desc="Epochs"):
            for step_idx in tqdm(range(0, args.n_steps), desc="Collecting Trajectory", leave=False):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                next_obs, actions, rewards, values, terminateds, log_probs = agent.buffer_prep(obs, envs, reward_list, terminateds)
                replay_buffer.store(obs, actions, rewards, values, terminateds, log_probs)

            with torch.no_grad():
                _, _, next_vals, _, _ = agent.get_logprob_and_value(next_obs)
                traj_adv, traj_ret = replay_buffer.calculate_advantages(next_vals.reshape(1, -1), next_terminateds.reshape(1, -1))

            traj_obs, traj_act, traj_val, traj_logprob = replay_buffer.get()
            train_batches(
                traj_obs.view(-1, *obs_dim), 
                traj_act, 
                traj_adv.view(-1), 
                traj_ret.view(-1), 
                traj_logprob.view(-1),
                agent, optimizer, scheduler, args
            )
            cum_reward = sum(reward_list)
            print('\n\n***************** Cumulative reward for this rollout: ', cum_reward, '*****************\n\n')
            reward_list = []
            # writer.add_scalar("charts/avg_reward", sum(reward_list), global_step_idx)
            # writer.add_scalar("losses/policy_loss", sum_loss_policy / args.train_iters, global_step_idx)
            # writer.add_scalar("losses/value_loss", sum_loss_value / args.train_iters, global_step_idx)
            # writer.add_scalar("losses/entropy", sum_entropy / args.train_iters, global_step_idx)
            # writer.add_scalar("losses/total_loss", sum_loss_total / args.train_iters, global_step_idx)
            # writer.add_scalar("charts/avg_reward", cum_reward, global_step_idx)
            # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step_idx)
            # writer.add_scalar("charts/SPS", global_step_idx / (time.time() - start_time), global_step_idx)

    finally:
            envs.close()
            test_env.close()
            writer.close()

            print("Test complete.")




# Main Function
if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

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

    agent, optimizer, scheduler, envs, test_env = initialize(args, device)

    start_time = time.time()

    training_loop(args, agent, envs, device, writer)