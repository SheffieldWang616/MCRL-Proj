import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
# from build_mlp import *
from buffer import Buffer


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, params):
        super(Agent, self).__init__()
        self.params = params

        self.obs_dim_size = self.params.obs_dim[0]
        self.act_dim_size = self.params.act_dim[0]
        
        self.obs_dim = self.params.obs_dim
        self.act_dim = self.params.act_dim
        
        # Actor Network for mu, Diagonal covariance matrix variables are separately trained
        # self.actor, self.actor_logstd = actor_net(self.obs_dim_size, self.act_dim_size)
        self.optimizer = optim.Adam(
                itertools.chain([self.actor_logstd], self.actor.parameters()),
                self.params.learning_rate
            )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=params.learning_rate_decay)
        
        # self.critic = critic_net(self.obs_dim_size)
        
        self.buffer = Buffer(self.obs_dim, self.act_dim, self.params.n_steps, self.params.n_envs, self.params.device, self.params.gamma, self.params.gae_lambda)
        self.device = self.params.device
        

    def forward(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.actor_logstd).expand_as(mu)

        return mu, std

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        mu, std = self.forward(obs)
        act_dist = torch.distributions.Normal(mu, std)
        # act_dist = self.forward(obs)
        if action is None:
            action = act_dist.sample()
        log_prob = act_dist.log_prob(action).sum(-1)
        entropy = act_dist.entropy().sum(-1)
        # entropy = act_dist.entropy()
        return action, log_prob, entropy, self.get_value(obs)

    def store_to_buffer(self, observation, envs, reward_list, terminateds, truncateds):
        # Sample the actions
        with torch.no_grad():
            actions, logprobs, _, values = self.get_action_and_value(observation)
            values = values.flatten()

        # Step the environment
        next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())
        # parse everything to tensors
        next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=self.device)
        reward_list.extend(rewards)
        rewards = torch.tensor(rewards, device=self.device).view(-1)
        next_terminateds = torch.tensor([float(term) for term in next_terminateds], device=self.device)
        next_truncateds = torch.tensor([float(trunc) for trunc in next_truncateds], device=self.device)

        # Store the step in the buffer
        self.buffer.store(observation, actions, rewards, values, terminateds, truncateds, logprobs)
        
    def sample_batch(self, next_obs, next_terminateds, next_truncateds):
        with torch.no_grad():
                # Finish the last step of the buffer with the value of the last state
                # and the terminated and truncated flags
                next_values = self.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = self.buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

        # Get the stored trajectories from the buffer
        traj_obs, traj_act, traj_val, traj_logprob = self.buffer.get()

        # Flatten the trajectories
        traj_obs = traj_obs.view(-1, *self.obs_dim)
        traj_act = traj_act.view(-1, *self.act_dim)
        traj_logprob = traj_logprob.view(-1)
        traj_adv = traj_adv.view(-1)
        traj_ret = traj_ret.view(-1)
        traj_val = traj_val.view(-1)

        # Create an array of indices to sample from the trajectories
        traj_indices = np.arange(self.params.n_steps * self.params.n_envs)
        
        return traj_obs, traj_act, traj_logprob, traj_adv, traj_ret, traj_val, traj_indices
    
    def update(self, traj_obs, traj_act, traj_logprob, traj_adv, traj_ret, traj_indices):
        sum_loss_policy = 0.0
        sum_loss_value = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0
        
        for _ in range(self.params.train_iters):
            # Shuffle the indices
            np.random.shuffle(traj_indices)
            
            for start_idx in range(0, self.params.n_steps, self.params.batch_size):
            
                end_idx = start_idx + self.params.batch_size
                batch_indices = traj_indices[start_idx:end_idx]

                # Get the log probabilities, entropies and values
                _, new_logprobs, entropies, new_values = self.get_action_and_value(traj_obs[batch_indices],
                                                                                    traj_act[batch_indices])
                ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])

                # normalize the advantages
                batch_adv = traj_adv[batch_indices]
                batch_adv = (batch_adv - batch_adv.mean()) / torch.max(batch_adv.std(),
                                                                        torch.tensor(1e-5, device=self.device))

                # Calculate the policy loss
                policy_loss1 = -batch_adv * ratios
                policy_loss2 = -batch_adv * torch.clamp(ratios, 1.0 - self.params.clip_ratio, 1.0 + self.params.clip_ratio)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Calculate the value loss
                new_values = new_values.view(-1)
                value_loss = 0.5 * ((new_values - traj_ret[batch_indices]) ** 2).mean()

                # Calculate the entropy loss
                entropy = entropies.mean()

                # Calculate the total loss
                loss = policy_loss + self.params.vf_coef * value_loss - self.params.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params.agent.parameters(), self.params.max_grad_norm)
                self.optimizer.step()

                sum_loss_policy += policy_loss.item()
                sum_loss_value += value_loss.item()
                sum_entropy += entropy.item()
                sum_loss_total += loss.item()
                
        self.scheduler.step()
        # return loss, policy_loss, value_loss, entropy
        return sum_loss_total, sum_loss_policy, sum_loss_value, sum_entropy

                
        