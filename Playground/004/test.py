import os
os.system('cls')
import gym
import minerl
from collections import OrderedDict
from MCagent import MineRLAgent
import pickle
from utils import *
'''
agent_parameters = pickle.load(open('md.model', "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
agent.load_weights('wt.weights')

env = gym.make("Tree-v0")
obs = env.reset()

done = False
while not done:
    ac, _ = agent.get_action(obs)
    obs, reward, done, info = env.step(ac)
    env.render()
env.close()

'''
agent_parameters = pickle.load(open('md.model', "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
agent.load_weights('wt.weights')

env = gym.make("Tree-v0")
obs = env.reset()['pov']

done = False
while not done:
    ac = agent.get_action(obs)
    # print(result['log_prob'].size())
    # print(result['pd']['buttons'].size())
    # print(result['pd']['camera'].size())
    obs, reward, done, info = env.step(ac)
    obs = obs['pov']
    env.render()
# log_video(env, agent, 'test.mp4')


env.close()


# print("AsyncVectorEnv Action Space:", env.action_space)
# print("AsyncVectorEnv Action Space Dict Keys:", env.action_space.spaces.keys())
# obs = env.reset()
# agent_parameters = pickle.load(open('md.model', "rb"))
# policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
# pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
# pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

# agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
# agent.load_weights('wt.weights')
# done = False
# for _ in range(10):
#     # ac = agent.get_action(obs)
#     # ac = env.action_space.noop()
#     # Spin around to see what is around us
#     # ac["camera"] = [0, 3]
#     print(ac)
#     obs, reward, done, info = env.step(ac)
#     # env.render()
# env.close()

# def create_action(num_envs):

#     actions_list = []
#     for i in range(num_envs):
#         action = env.action_space.noop()
#         action["camera"] = [0, 3]
#         actions_list.append(action)

#     merged_actions = OrderedDict({
#         key: [action[key] for action in actions_list]
#         for key in actions_list[0].keys()
#     })

#     return merged_actions

# ac = create_action(2)
# print(ac)