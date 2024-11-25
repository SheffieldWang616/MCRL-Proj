import pickle
from MCagent import MineRLAgent
import torch


'''Model path and weights path'''
# model_path = 'md.model'
# weights_path = 'wt.weights'

model_path = "Model_Weights/2x_pre/2x.model"
weights_path = "Model_Weights/2x_pre/foundation-model-2x.weights"

with open(model_path, 'rb') as f:
    agent_parameters = pickle.load(f)
    
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]

agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
model = agent.policy
agent.load_weights(weights_path)

with open('all_obs.pkl', 'rb') as f:
    obs = pickle.load(f)
# obs is saved from buffer, designed shape is (batch, n_env, 360, 640, 3), here is (1024, 1, 360, 640, 3)
obs = obs.squeeze(1)

# Change batch to test, 1 won't error, more would
batch = 3
# agent_input is a dict, key is 'img', value is a tensor with shape (batch, 128, 128, 3), use smaller batch (2 - 64) to avoid memory error
agent_input = agent._env_obs_to_agent(obs[:batch]) 

print('Obs Batch Size: ', agent_input['img'].shape)
'''
This is getting probability distribution of action from current parameters to get log prob of past actions, in order to calculate ratio
If agent input is 1 single batch only, no mistake, but if it is multiple batch, it will raise error
There are two potential error, one is in lib.xf, around line 386, full = th.cat([prev[:, startfull:], new], dim=1), 
the prev and new has different shape, seems like the batch and time dimension is not on the same dimension, so can't cat
If permute the dimentions, it can cat, but in the same file, around line 56, the bias = bias + extra_btT would raise error,
bias是size(16, 1, 129), 但是extra_btT的size是(16*batch, 1, 129 + batch - 1), 只有batch是1的时候extra_btT和bias的size是一样的
'''
pd, vpred, _ = agent.policy.get_output_for_observation(agent_input, agent.hidden_state, agent._dummy_first)

print("Camera Probability Distribution: ", pd['camera'].shape)
print("Buttons Probability Distribution: ", pd['buttons'].shape)
print("Value Prediction: ", vpred.shape)
print("Value Prediction = \n", vpred)