import os
os.system('cls')
import numpy as np
import torch
import minerl
import pickle
from MCagent import MineRLAgent
import gym
from utils import *
import multiprocessing
from gym.vector import AsyncVectorEnv

# agent_parameters = pickle.load(open('./Model_Weights/2x_pre/2x.model', "rb"))
# policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
# pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
# pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
# agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
# agent.load_weights('./Model_Weights/2x_pre/2x.weights')

# n_env = 2
# # env = gym.make("Tree-v0")
# envs = gym.vector.AsyncVectorEnv([lambda: make_env("Tree-v0") for _ in range(n_env)])
# obs = envs.reset()

# print(obs)

if __name__ == '__main__':


    agent_parameters = pickle.load(open('./Model_Weights/2x_pre/2x.model', "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights('./Model_Weights/2x_pre/2x.weights')

    n_env = 2
    # env = gym.make("Tree-v0")
    envs = gym.vector.AsyncVectorEnv([lambda: make_env("Tree-v0") for _ in range(n_env)])

    try:
        obs = envs.reset()

        # actions = [agent.get_action(obs[i]) for i in range(n_env)]
        # next_obs, rewards, dones, infos = envs.step(actions)
        print("Observation type:", type(obs))
        for i in len(obs[:]):
            print("Observation shape:",np.shape(obs[i]))
        # print("Observation shape:", np.shape(ob))
        # print("obs_pov", obs[:]['pov'])

    
    finally:
        envs.close()