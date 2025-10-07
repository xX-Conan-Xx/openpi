import torch
import torch.nn as nn
import robomimic.utils.tensor_utils as TensorUtils

from .base_policy import BasePolicy
from .modules.mlp import DynamicMLP
from .modules.rgb_modules import *
from .modules.language_modules import *
from .modules.transformer_modules import *
from .modules.policy_head import *
from .bc_transformer_policy import ExtraModalityTokens

###############################################################################
#
# An MLP Policy
#
###############################################################################


class BCMLPPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        embed_size = policy_cfg.embed_size
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(      # 2.563904 M
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(      # 0.049216 M
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(       # 0.000704 M
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_state_encoder.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_state_encoder.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        ### 4. define spatial mlp
        self.spatial_down_sample = nn.Linear(embed_size * cfg.data.seq_len * 5, policy_cfg.spatial_mlp.input_size)  # 0.204864M
        spatial_mlp_size = [policy_cfg.spatial_mlp.input_size] + policy_cfg.spatial_mlp.hidden_size + [policy_cfg.spatial_mlp.output_size]
        self.spatial_mlp = DynamicMLP(spatial_mlp_size)     # 0.099904 M

        ### 5. define policy head
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(    # 1.123335 M
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )

        self.latent_queue = []
        self.max_seq_len = cfg.data.seq_len

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded = [text_encoded, extra]

        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data, return_latent=False):
        x = self.spatial_encode(data)
        x = TensorUtils.join_dimensions(x, 2, 3)  # (B, T, num_modality*E)
        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality*E)
        x = self.spatial_down_sample(x)
        z = self.spatial_mlp(x)
        dist = self.policy_head(z)

        if return_latent:
            return x, z, dist
        
        return dist

    def get_action(self, cfg, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)
            self.latent_queue.append(x)
            if len(self.latent_queue) < self.max_seq_len:
                for i in range(self.max_seq_len - len(self.latent_queue)):
                    self.latent_queue.append(x)
            if len(self.latent_queue) > self.max_seq_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)  # (B, T, num_modality, E)
            x = TensorUtils.join_dimensions(x, 2, 3)  # (B, T, num_modality*E)
            x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality*E)
            x = self.spatial_down_sample(x)
            x = self.spatial_mlp(x)

        if cfg.policy.policy_head.network == 'GMMHead':    
            dist = self.policy_head(x)
            action = dist.sample().detach().cpu()
            return action.view(action.shape[0], -1).numpy()
        elif cfg.policy.policy_head.network == 'DeterministicHead':
            action = self.policy_head(x)
            action = action.detach().cpu()
            action = torch.clamp(action, -1, 1)
            return action.float().numpy()
        else:
            raise ValueError('The policy head is set incorrectly.')

    def reset(self):
        self.latent_queue = []
