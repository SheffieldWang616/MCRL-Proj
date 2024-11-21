import gym
import minerl
import os
import datetime
import numpy as np

env = gym.make("Tree-v0")
obs = env.reset()

done = False
saved_obs = []
save_dir = './Model_Weights/PPO'
os.makedirs(save_dir, exist_ok=True)
print(f'obs saved to {save_dir}')

file_name = datetime.datetime.now().strftime('Test_obs_%Y-%m-%d_%H-%M-%S.npy')
for _ in range(100):
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 3]
    print(ac)
    obs, reward, done, info = env.step(ac)
    # env.render()
    
    saved_obs.append(obs['pov'])

    # save the observation under save_dir with file_name
    np.save(os.path.join(save_dir, file_name), np.array(saved_obs))


env.close()