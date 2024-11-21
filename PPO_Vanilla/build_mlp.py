import torch
import numpy as np

def actor_net(obs_dim, act_dim):
    actor_hid1_size = obs_dim * 10
    actor_hid3_size = act_dim * 10
    actor_hid2_size = int(np.sqrt(actor_hid1_size * actor_hid3_size))
    actor_mu = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, actor_hid1_size),
        torch.nn.Tanh(),
        torch.nn.Linear(actor_hid1_size, actor_hid2_size),
        torch.nn.Tanh(),
        torch.nn.Linear(actor_hid2_size, actor_hid3_size),
        torch.nn.Tanh(),
        torch.nn.Linear(actor_hid3_size, act_dim),
        torch.nn.Tanh()
    )
    actor_logstd = torch.nn.Parameter(torch.zeros(act_dim))
    
    return actor_mu, actor_logstd

def critic_net(obs_dim):
    critic_hid1_size = obs_dim * 10
    critic_hid3_size = 5
    critic_hid2_size = int(np.sqrt(critic_hid1_size * critic_hid3_size))
    critic = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, critic_hid1_size),
        torch.nn.Tanh(),
        torch.nn.Linear(critic_hid1_size, critic_hid2_size),
        torch.nn.Tanh(),
        torch.nn.Linear(critic_hid2_size, critic_hid3_size),
        torch.nn.Tanh(),
        torch.nn.Linear(critic_hid3_size, 1)
    )
    
    return critic