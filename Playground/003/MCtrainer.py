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
    parser.add_argument("--n-epochs", type=int, default=1, help="Number of epochs to run") # 3000 is fully trained
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
    parser.add_argument("--weights", type=str, default = 'F:\\16831_RL\\Proj\\MCRL-Proj\\Model_Weights\\2x_pre\\rl-from-house-2x.weights', help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default = 'F:\\16831_RL\\Proj\\MCRL-Proj\\Model_Weights\\2x_pre\\2x.model', help="Path to the '.model' file to be loaded.")
    
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
    agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.weights)
    
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
    replay_buffer = Buffer(next_obs.shape,args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, agent=agent)    
    next_obs = torch.tensor(np.array(next_obs, dtype=np.uint8), device=device)
    next_terminateds = torch.tensor([float(False)], device=device)
    
    reward_list = []
    
    try:
        for epoch in range(1, args.n_epochs + 1):
            for step_idx in range(0, args.n_steps):
                obs = next_obs
                terminateds = next_terminateds
                next_obs, actions, rewards, values, terminateds, log_probs = agent.buffer_prep(obs, envs, reward_list, terminateds)
                # print(type(obs),type(next_obs), type(actions), type(rewards), type(values), type(terminateds), type(log_probs))
                # print(np.shape(obs))
                # # print(f"Strides: {observation.strides}")
                # observation = torch.tensor(obs.copy(), device=device)
                # print(type(observation))
                # print(observation.shape)
                # print(f"Strides: {observation.strides}")
                # break
                replay_buffer.store(obs, actions, rewards, values, terminateds, log_probs)
            break
        # print(actions)
        # for epoch in range(1, args.n_epochs + 1):
        #     # Collect trajectories
        #     for step_idx in range(0, args.n_steps):
        #         global_step_idx += args.n_envs
        #         obs = next_obs
        #         terminateds = next_terminateds
        #         truncateds = next_truncateds
        #         agent.buffer_prep(obs, envs, reward_list, terminateds, truncateds)
        #         TODO: replay_buffer.store(observation, actions, rewards, values, terminateds, log_probs)
                
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