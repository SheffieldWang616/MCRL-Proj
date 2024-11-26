import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces
import torchvision.transforms as T

from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy
from lib.torch_util import default_device_type, set_default_torch_device
from tqdm import tqdm


# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def validate_env(env):
    """Check that the MineRL environment is setup correctly, and raise if not"""
    for key, value in ENV_KWARGS.items():
        if key == "frameskip":
            continue
        if getattr(env.task, key) != value:
            raise ValueError(f"MineRL environment setting {key} does not match {value}")
    action_names = set(env.action_space.spaces.keys())
    if action_names != set(TARGET_ACTION_SPACE.keys()):
        raise ValueError(f"MineRL action space does match. Expected actions {set(TARGET_ACTION_SPACE.keys())}")

    for ac_space_name, ac_space_space in TARGET_ACTION_SPACE.items():
        if env.action_space.spaces[ac_space_name] != ac_space_space:
            raise ValueError(f"MineRL action space setting {ac_space_name} does not match {ac_space_space}")


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    print('+++++++++++++++ obs before reshape', img.shape,'+++++++++++++++')
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


class MineRLAgent:
    def __init__(self, device=None, policy_kwargs=None, pi_head_kwargs=None):
        # validate_env(env)

        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)
        
        self.log_prob_and_v = None
        self.non_transformed_action = None
        
        # self.buffer = Buffer()

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)
        
        

    # def _env_obs_to_agent(self, minerl_obs):
    #     """
    #     Turn observation from MineRL environment into model's observation

    #     Returns torch tensors.
    #     """
    #     agent_input = resize_image(minerl_obs, AGENT_RESOLUTION)[None]
    #     print('+++++++++++++++ obs after reshape', agent_input.shape,'+++++++++++++++')
    #     agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
    #     return agent_input
    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation.

        Handles:
        - Single observation: ndarray (360, 640, 3), resized to (1, 128, 128, 3)
        - Batch of observations: tensor (n, 360, 640, 3), resized to (n, 128, 128, 3)

        Returns torch tensors.
        """
        # Check if input is a NumPy array (single observation)
        if isinstance(minerl_obs, np.ndarray):
            # Ensure contiguous memory layout
            minerl_obs = minerl_obs.copy()
            
            # Convert NumPy array to PyTorch tensor and reshape
            minerl_obs_tensor = th.from_numpy(minerl_obs).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
            
            # Resize using torchvision transforms
            transform = T.Resize((128, 128))
            resized_obs = transform(minerl_obs_tensor)
            
            # Convert back to channel-last format and add batch dimension
            resized_obs = resized_obs.permute(1, 2, 0).unsqueeze(0)  # (C, H, W) -> (1, H, W, C)
        
        # Check if input is a PyTorch tensor (batch of observations)
        elif isinstance(minerl_obs, th.Tensor):
            # Ensure input is in the expected format (n, H, W, C)
            # if minerl_obs.dim() != 4 or minerl_obs.size(-1) != 3:
            #     raise ValueError("Expected input tensor with shape (n, 360, 640, 3).")
            if minerl_obs.dim() != 4:
                minerl_obs = minerl_obs.unsqueeze(0)
            
            # Permute to (n, C, H, W) for torchvision.transforms
            minerl_obs_tensor = minerl_obs.permute(0, 3, 1, 2).float()  # (n, H, W, C) -> (n, C, H, W)
            
            # Resize each image in the batch using transforms
            transform = T.Resize((128, 128))
            resized_obs = th.stack([transform(img) for img in minerl_obs_tensor])  # Resize each batch individually
            
            # Convert back to channel-last format
            resized_obs = resized_obs.permute(0, 2, 3, 1)  # (n, C, H, W) -> (n, H, W, C)
        
        else:
            raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
        
        # Send to the desired device and return as dictionary
        agent_input = {"img": resized_obs.to(self.device)}
        # print('+++++++++++++++ obs after reshape', resized_obs.size(), '+++++++++++++++')
        return agent_input
    
    

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action

    def get_action(self, minerl_obs):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, results = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True,
        )
        # print('Action:', agent_action)
        minerl_action = self._agent_action_to_env(agent_action)
        self.log_prob_and_v = results
        # self.log_prob_and_v['original_action'] = agent_action
        
        return minerl_action

    def get_logprob_and_value(self, obs, action = None):
        minerl_action = None
        entropy = None
        if action is None:
            minerl_action = self.get_action(obs)
            action = self.log_prob_and_v['raw_action']
            log_prob = self.log_prob_and_v['log_prob']
            value = self.log_prob_and_v['vpred']
        else:
            # with th.no_grad():
                agent_input = [self._env_obs_to_agent(ob) for ob in obs]
                
                batch_size = len(agent_input)
                
                log_prob = th.empty(batch_size, device=self.device)  # 1D tensor for log_probs
                value = th.empty(batch_size, device=self.device)
                entropy = th.empty(batch_size, device=self.device)
                
                for i in tqdm(range(len(agent_input)), desc='Getting log_probs and values from trajectory'):
                    pd, vpred, _ = self.policy.get_output_for_observation(agent_input[i], self.hidden_state, self._dummy_first)
                    lgp = self.policy.get_logprob_of_action(pd, action[i])
                    log_prob[i] = lgp
                    value[i] = vpred
                    entropy[i] = self.policy.pi_head.entropy(pd)
                    # print('\n\n', type(lgp), lgp, type(vpred), vpred, '\n\n')
                    # print('\n\n', self.policy.pi_head.entropy(pd), '\n\n')
                    # break
            # log_prob = th.stack(log_prob)
        
        return action, log_prob, value, minerl_action, entropy
    
    def buffer_prep(self, observation, envs, reward_list, terminateds):
        
        actions, log_probs, values, minerl_action, _ = self.get_logprob_and_value(observation)
        values = values.flatten()
        
        next_obs, rewards, next_terminateds, _ = envs.step(minerl_action)
        next_obs = th.tensor(np.array(next_obs['pov'], dtype=np.uint8), device=self.device)
        if isinstance(rewards, list):
            reward_list.extend(rewards)
            rewards = th.tensor(rewards, device=self.device).view(-1)
            next_terminateds = th.tensor([float(term) for term in next_terminateds], device=self.device)
        else:
            reward_list.append(rewards)
            rewards = th.tensor(rewards, device=self.device).view(-1)
            next_terminateds = th.tensor([float(next_terminateds)], device=self.device)
        
        return next_obs, actions, rewards, values, terminateds, log_probs
        
    
    