import os
# os.add_dll_directory("C://Users//yhttmb//.mujoco//mujoco210//bin")
os.system('cls')
import argparse, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import minerl
# from collections import OrderedDict
import pickle
# from multiprocessing import Process, Pipe
# from stable_baselines3.common.vec_env import SubprocVecEnv

from MCagent import MineRLAgent
from buffer import Buffer
from utils import *
from tqdm import tqdm

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
    parser.add_argument("--env", default="Tree-v0", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments") # Parallel
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
    parser.add_argument("--render-epoch", type=int, default=200, help="Render every n-th epoch")
    parser.add_argument("--save-epoch", type=int, default=200, help="Save the model every n-th epoch")
    # NOTE: add checkpoint path as argument
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to resume checkpoint")
    # NOTE: Existing model
    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    
    return parser.parse_args()

# def env_worker():

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

    agent_parameters = pickle.load(open(args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(device = args.device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.weights)
    
    optimizer = optim.Adam(agent.policy.parameters(), lr=args.learning_rate, eps=1e-5)
    # optimizer = optim.SGD(agent.policy.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    
    envs = gym.make(args.env)
    test_env = gym.make(args.env)
    # TODO Build buffer
    # replay_buffer = Buffer(args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, agent=agent)
    # start_epoch = 1
    # if args.checkpoint_path and os.path.exists(args.checkpoint_path):
    #     start_epoch = load_checkpoint(agent, optimizer, args.checkpoint_path)
    
    global_step_idx = 0
    start_time = time.time()
    init_obs = envs.reset()
    next_obs = init_obs['pov']
    obs_dim = next_obs.shape
    replay_buffer = Buffer(obs_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, agent=agent)   
    next_obs = torch.tensor(np.array(next_obs, dtype=np.uint8), device=device)
    next_terminateds = torch.tensor([float(False)], device=device)
    
    reward_list = []
    
    try:
        for epoch in tqdm(range(1, args.n_epochs + 1), desc="Epochs"):
            for step_idx in tqdm(range(0, args.n_steps), desc=f"Collecting Trajectory", leave=False):
                obs = next_obs
                terminateds = next_terminateds
                next_obs, actions, rewards, values, terminateds, log_probs = agent.buffer_prep(obs, envs, reward_list, terminateds)
                replay_buffer.store(obs, actions, rewards, values, terminateds, log_probs)
                # print(replay_buffer.obs_buf.size())
                # break
            # break
            # print(replay_buffer.act_buf)
            with open('action_buffer.pkl', 'wb') as f:
                pickle.dump(replay_buffer.act_buf, f)
            # print(replay_buffer.val_buf.size())
            with torch.no_grad():
                _, _, next_vals, _, _ = agent.get_logprob_and_value(next_obs)
                next_vals = next_vals.reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                traj_adv, traj_ret = replay_buffer.calculate_advantages(next_vals, next_terminateds)
            
            traj_obs, traj_act, traj_val, traj_logprob = replay_buffer.get()
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)
            traj_val = traj_val.view(-1)
            
            # print(type(traj_obs), type(traj_act), type(traj_val), type(traj_logprob), type(traj_adv), type(traj_ret))
            
            traj_indices = np.arange(args.n_steps * args.n_envs)
            
            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_total = 0.0
            for _ in range(args.train_iters):

                np.random.shuffle(traj_indices)

                for start_idx in range(0, args.n_steps, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx].tolist()
                    
                    _, new_logprobs, new_values, _, entropies = agent.get_logprob_and_value(traj_obs[batch_indices], traj_act[batch_indices])
                    
                    ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])
                    batch_adv = traj_adv[batch_indices]
                    batch_adv = (batch_adv - batch_adv.mean()) / torch.max(batch_adv.std(), torch.tensor(1e-5, device=device))
                    # print(entropies.size(), ratios.size(), batch_adv.size())
                    
                    policy_loss1 = -batch_adv * ratios
                    policy_loss2 = -batch_adv * torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                    value_loss = 0.5 * ((new_values - traj_ret[batch_indices]) ** 2).mean()

                    entropy = entropies.mean()
                    
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                    
                    # optimizer.zero_grad()
                    # loss.backward()
                    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    # optimizer.step()

                    # sum_loss_policy += policy_loss.item()
                    # sum_loss_value += value_loss.item()
                    # sum_entropy += entropy.item()
                    # sum_loss_total += loss.item()
                    
                    break
                break
            
            break
                
    finally:
        envs.close()
        test_env.close()
        writer.close()
        print('Test complete.')
        
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # envs = gym.vector.AsyncVectorEnv([lambda: make_env(args.env) for _ in range(args.n_envs)])
    # assert envs.num_envs == args.n_envs, f"Expected {args.n_envs} environments, got {envs.num_envs}"
    # env = 
    # print(f"{args.n_envs} environments created successfully.")
    
    # observations = envs.reset()
    
    # batch_action = agent.get_action(observations)
    # print(batch_action)
    # envs = SubprocVecEnv([make_env(args.env) for _ in range(args.n_envs)])
    # envs = DummyVecEnv([make_env(args.env) for _ in range(args.n_envs)])
    # test_env = gym.make(args.env)
    # print(f"{args.n_envs} environments created successfully.")
    
    # obs = envs.reset()

    # all_observations = []
    # all_observations.append(observations['pov'])
    # # minerl_action = [agent.get_action(obs) for obs in unpack_obs(observations)]
    
    # for _ in tqdm(range(1000), desc="Collecting observations"):
    #     # start_time = time.time()
    #     minerl_action = [agent.get_action(obs) for obs in unpack_obs(observations)]
    #     # action_time = time.time() - start_time
        
    #     observations, rewards, dones, infos = envs.step(minerl_action)
    #     # step_time = time.time() - start_time - action_time
        
    #     # print(f"Action time: {action_time:.4f}s, Step time: {step_time:.4f}s")
        
    #     all_observations.append(observations['pov'])

    # with open('all_observations.pkl', 'wb') as f:
    #     pickle.dump(all_observations, f)
    # actions = [agent.get_action(obs) for obs in unpack_obs(observations)]
    # print("Actions:", actions)
    # obs, reward, dones, info = envs.step(actions)
    # print('Successfully stepped through environments')
    
    
    # with open('all_observations.pkl', 'wb') as f:
    #     pickle.dump(observations, f)
    # print("Initial Observations:", observations)
    # action_space = envs.single_action_space.no_op()

    # all_observations = []
    # for _ in tqdm(range(1000), desc="Collecting observations"):
    #     minerl_action = get_all_actions(obs, agent=agent, action_space=action_space)
    #     obs, reward, dones, info = envs.step(minerl_action)
    #     all_observations.append(obs['pov'])
    #     if any(dones):
    #         envs.reset_done()
    # with open('all_observations.pkl', 'wb') as f:
    #     pickle.dump(all_observations, f)
    
    '''
    action_space = envs.single_action_space.no_op()
    minerl_action = get_all_actions(observations, agent=agent, action_space=action_space)

    all_observations = []
    for _ in range(1000):
        minerl_action = get_all_actions(observations, agent=agent, action_space=action_space)
        obs, reward, dones, info = envs.step(minerl_action)
        all_observations.append(obs)
        if any(dones):
            envs.reset_done()
    with open('all_observations.pkl', 'wb') as f:
        pickle.dump(all_observations, f)
    '''