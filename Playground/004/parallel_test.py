import os
os.system('cls')
import numpy as np
import torch
import minerl
import pickle
from MCagent import MineRLAgent
import gym
from utils import *

agent_parameters = pickle.load(open('md.model', "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
agent.load_weights('wt.weights')

n_env = 2
# env = gym.make("Tree-v0")
envs = gym.vector.AsyncVectorEnv([lambda: make_env("Tree-v0") for _ in range(n_env)])
obs = envs.reset()

print(obs)