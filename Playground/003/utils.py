import torch
import cv2
import numpy as np
import gym
from collections import OrderedDict

def log_video(env, agent, device, video_path, fps=30):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        # Render the frame
        frames.append(env.render())
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        obs, _, terminated, _, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated
    # Save the video
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# def make_env(env_name):
#     def _init():
#         # Ensure each process creates a unique environment instance
#         return gym.make(env_name)
#     return _init

# def unpack_obs(batched_obs):
#     keys = batched_obs.keys()
#     n_envs = len(batched_obs['inventory'])  # Get the number of environments
#     list_of_obs = [{key: batched_obs[key][i] for key in keys} for i in range(n_envs)]
#     return list_of_obs

# def get_all_actions(all_obs, agent):
#     return [agent.get_action(obs) for obs in unpack_obs(all_obs)]



def make_env(env_id, render=False, fps=30):
    """
    Make an environment with the given id.
    :param env_id: the id of the environment.
    :param reward_scaling: the scaling factor for the rewards.
    :param render: whether to render the environment.
    :param fps: the frames per second if rendering.
    :return: the environment.
    """
    # if render:
    #     env = gym.make(env_id, render_mode='rgb_array')
    #     env.metadata['render_fps'] = fps
    #     # env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    # else:
    #     env = gym.make(env_id)
    #     # env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    
    env = gym.make(env_id)
    return env

# def create_action():
#     action = envs.single_action_space.noop()
#     # action = envs.action_space.noop()
#     action['camera'] = [0, 3]  # Rotate camera to the right
#     return action

# def merge_action(individual_actions, single_action_space):
#     """
#     Transform a list of individual actions into a batched action format for AsyncVectorEnv.

#     Args:
#         individual_actions (list[dict]): List of individual actions from the model.
#         single_action_space: The single action space of the environment (to provide default keys/values).

#     Returns:
#         OrderedDict: Batched actions in the format required by AsyncVectorEnv.
#     """
#     # Get the default action structure
#     default_action = single_action_space

#     # Initialize the batched action dictionary
#     batched_actions = OrderedDict({key: [] for key in default_action.keys()})

#     # Fill batched actions for each environment
#     for individual_action in individual_actions:
#         for key in batched_actions.keys():
#             if key in individual_action:
#                 # Handle 'camera' key specifically
#                 if key == 'camera':
#                     value = individual_action[key]
#                     if isinstance(value, np.ndarray):
#                         batched_actions[key].append(value.flatten().tolist())  # Flatten nested arrays
#                     elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
#                         batched_actions[key].append(value[0])  # Handle lists of lists
#                     else:
#                         batched_actions[key].append(value)  # Use as-is for other valid formats
#                 else:
#                     # General handling for other keys
#                     value = individual_action[key]
#                     if isinstance(value, np.ndarray) and value.size == 1:
#                         batched_actions[key].append(value.item())  # Convert single-element array to scalar
#                     elif isinstance(value, np.ndarray):
#                         batched_actions[key].append(value.tolist())  # Convert array to list
#                     else:
#                         batched_actions[key].append(value)  # Use as-is
#             else:
#                 # Use default value if key is missing
#                 batched_actions[key].append(default_action[key])

#     return batched_actions


# def split_obs(obs):
#     num_envs = obs["pov"].shape[0]
#     obervations = []
#     for i in range(num_envs):
#         single_observation = OrderedDict({
#             "inventory": {key: value[i] for key, value in obs["inventory"].items()},
#             "pov": obs["pov"][i]
#         })
#         obervations.append(single_observation)
#     return obervations

# def get_all_actions(obs, agent, action_space):
#     obs_list = split_obs(obs)
#     actions_list = []
#     for obs in obs_list:
#         actions_list.append(agent.get_action(obs))
#     actions = merge_action(actions_list, action_space)

#     return actions
