import torch
import cv2
import numpy as np
import gymnasium as gym

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
    
def make_env(env_id, reward_scaling=1.0, render=False, fps=30):
    """
    Make an environment with the given id.
    :param env_id: the id of the environment.
    :param reward_scaling: the scaling factor for the rewards.
    :param render: whether to render the environment.
    :param fps: the frames per second if rendering.
    :return: the environment.
    """
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env



# import gym
# import minerl

# # Uncomment to see more logs of the MineRL launch
# # import coloredlogs
# # coloredlogs.install(logging.DEBUG)

# env = gym.make("MineRLBasaltBuildVillageHouse-v0")
# obs = env.reset()

# done = False
# while not done:
#     ac = env.action_space.noop()
#     # Spin around to see what is around us
#     ac["camera"] = [0, 3]
#     obs, reward, done, info = env.step(ac)
#     env.render()
# env.close()