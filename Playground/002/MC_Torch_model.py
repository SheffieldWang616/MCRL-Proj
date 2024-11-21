import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomIMPALA(nn.Module):
    def __init__(self, input_shape, chans, width):
        super(CustomIMPALA, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[2]  # Number of input channels (e.g., 3 for RGB)

        for out_channels in chans:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_channels = out_channels

        self.fc = nn.Linear(chans[-1] * (input_shape[0] // (2 ** len(chans))) * (input_shape[1] // (2 ** len(chans))), width)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        return self.fc(x)


class CustomTransformer(nn.Module):
    def __init__(self, input_dim, hidsize, num_heads, num_layers, memory_size):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidsize)
        self.positional_encoding = nn.Parameter(torch.randn(1, memory_size, hidsize))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidsize,
                nhead=num_heads,
                dim_feedforward=4 * hidsize,
                activation="relu"
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x


class ReconstructedModel(nn.Module):
    def __init__(self, policy_kwargs):
        super(ReconstructedModel, self).__init__()
        img_shape = policy_kwargs["img_shape"]
        impala_chans = policy_kwargs["impala_chans"]
        impala_width = policy_kwargs["impala_width"]
        hidsize = policy_kwargs["hidsize"]
        attention_heads = policy_kwargs["attention_heads"]
        attention_memory_size = policy_kwargs["attention_memory_size"]
        n_recurrence_layers = policy_kwargs["n_recurrence_layers"]

        self.impala = CustomIMPALA(img_shape, impala_chans, impala_width)
        self.recurrence = CustomTransformer(
            input_dim=impala_width,
            hidsize=hidsize,
            num_heads=attention_heads,
            num_layers=n_recurrence_layers,
            memory_size=attention_memory_size
        )
        self.policy_head = nn.Linear(hidsize, 10)  # Example output size, adjust if necessary
        self.value_head = nn.Linear(hidsize, 1)

    def forward(self, obs):
        x = self.impala(obs["pov"])
        x = self.recurrence(x.unsqueeze(1))  # Add time dimension
        x = x[:, -1, :]  # Use the last timestep
        return self.policy_head(x), self.value_head(x)

    def share_memory(self):
        self.impala.share_memory()
        self.recurrence.share_memory()
        self.policy_head.share_memory()
        self.value_head.share_memory()

















#----------------------------------------- Model using ./video-pre-training/lib -----------------------------------------
# import torch
# import torch.nn as nn
# from lib.policy import MinecraftPolicy
# from lib.actions import Buttons, CameraQuantizer
# from lib.action_mapping import ActionMapping
# from lib.action_head import ActionHead
# import pickle

# class ReconstructedModel(nn.Module):
#     def __init__(self, policy_kwargs, pi_head_kwargs, action_mapping_kwargs, camera_binsize=11, camera_maxval = 10.0):
#         """
#         Reconstructed policy model with actions integrated from lib files.
#         Args:
#             policy_kwargs (dict): Arguments for the MinecraftPolicy backbone.
#             pi_head_kwargs (dict): Arguments for the policy head.
#             action_mapping_kwargs (dict): Arguments for action mapping.
#             camera_binsize (int): Number of bins for discretizing camera movements.
#         """
#         super().__init__()

#         # Initialize MinecraftPolicy as the backbone
#         self.net = MinecraftPolicy(**policy_kwargs)

#         # Initialize CameraQuantizer for continuous actions
#         self.camera_quantizer = CameraQuantizer(camera_maxval=camera_maxval, camera_binsize=camera_binsize)

#         # Buttons and camera mapping
#         self.buttons = Buttons.ALL
#         self.action_mapping = ActionMapping(n_camera_bins=camera_binsize, **action_mapping_kwargs)

#         # Action and value heads
#         self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)
#         self.value_head = nn.Linear(self.net.output_latent_size(), 1)

#     def make_action_head(self, input_dim, **pi_head_opts):
#         """
#         Creates the action head using ActionHead from lib.action_head.
#         Args:
#             input_dim (int): Dimension of the input to the action head.
#         Returns:
#             nn.Module: The action head module.
#         """
#         return ActionHead(input_dim=input_dim, **pi_head_opts)

#     def forward(self, obs, first, state_in):
#         """
#         Forward pass through the model.
#         Args:
#             obs (dict): Observations from the environment.
#             first (torch.Tensor): Episode start indicator.
#             state_in (Any): Initial state for the recurrent layers.
#         Returns:
#             tuple: Action logits, value prediction, and updated recurrent state.
#         """
#         # Process observations with the policy backbone
#         (pi_h, v_h), state_out = self.net(obs, state_in, context={"first": first})

#         # Action logits and value prediction
#         pi_logits = self.pi_head(pi_h)
#         vpred = self.value_head(v_h)

#         return (pi_logits, vpred), state_out

#     def initial_state(self, batch_size):
#         """
#         Returns the initial recurrent state.
#         Args:
#             batch_size (int): Batch size.
#         Returns:
#             Any: Initial state.
#         """
#         return self.net.initial_state(batch_size)

#     def map_action(self, action_logits):
#         """
#         Maps logits into the environment's action space.
#         Args:
#             action_logits (torch.Tensor): Raw action logits.
#         Returns:
#             dict: Mapped actions (buttons and camera movements).
#         """
#         # Discretize camera movements and map button logits to actions
#         button_actions = self.action_mapping.map_buttons(action_logits)
#         camera_actions = self.camera_quantizer.discretize(action_logits["camera"])

#         return {"buttons": button_actions, "camera": camera_actions}

#     def share_memory(self):
#         """
#         Makes the model multiprocessing-compatible by sharing memory.
#         """
#         self.net.share_memory()
#         self.pi_head.share_memory()
#         self.value_head.share_memory()


# # **Load Pretrained Weights**
# def load_pretrained_weights(model, model_path, weights_path):
#     """
#     Load weights from `.model` and `.weights` files into the ReconstructedModel.
#     Args:
#         model (ReconstructedModel): The model to load weights into.
#         model_path (str): Path to the `.model` file.
#         weights_path (str): Path to the `.weights` file.
#     """
#     # Load the model file for configurations
#     with open(model_path, "rb") as f:
#         model_data = pickle.load(f)
#     policy_args = model_data["model"]["args"]["net"]["args"]
#     pi_head_opts = model_data["model"]["args"]["pi_head_opts"]

#     # Ensure the model matches the configuration
#     model.net.load_policy_config(policy_args)
#     model.pi_head.load_pi_head_config(pi_head_opts)

#     # Load weights file
#     model.load_state_dict(torch.load(weights_path))
#     print("Weights loaded successfully.")
