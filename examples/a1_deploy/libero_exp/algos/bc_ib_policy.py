import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseAlgo
from ..utils.mine_utils import Mine


class BC_IB_Policy(BaseAlgo):
    def __init__(self, cfg, inference=False, device='cuda'):
        super().__init__(cfg, inference, device)
        if not inference:
            if cfg.policy.policy_type == 'BCRNNPolicy':
                input_dim = cfg.policy.rnn_hidden_size * cfg.data.seq_len
            elif cfg.policy.policy_type == 'BCTransformerPolicy' or cfg.policy.policy_type == 'BCDPPolicy':
                input_dim = cfg.policy.temporal_transformer.transformer_head_output_size * cfg.data.seq_len * 5
            elif cfg.policy.policy_type == 'BCViLTPolicy':
                input_dim = cfg.policy.temporal_transformer.transformer_head_output_size * cfg.data.seq_len * 4
            elif cfg.policy.policy_type == 'BCMLPPolicy':
                input_dim = cfg.policy.embed_size
            hidden_dim = 400
            mine_model =  nn.Sequential(
                nn.Linear(input_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.mine = Mine(
                model = mine_model,
                loss_type = 'mine',  # mine_biased, fdiv
            )
            self.mine.model = self.mine.model.to(self.device)
            self.mine_optimizer = torch.optim.Adam(self.mine.model.parameters(), lr=1e-5)

    def forward_backward(self, data):
        data = self.model.preprocess_input(data)
        bc_loss, mi_loss = self.compute_loss(data)
        loss = bc_loss + mi_loss

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        mi_loss_2 = self.compute_mine_model_loss(data)
        self.mine_optimizer.zero_grad()
        mi_loss_2.backward()
        self.mine_optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "mi_loss": mi_loss.item(),
            "mine": -mi_loss_2.item(),
        }

        return ret_dict
    
    def compute_loss(self, data):
        x, z, dist = self.model(data, return_latent=True)
        if self.cfg.policy.policy_type == 'BCMLPPolicy':
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"][:, -1], reduction="mean")
        elif self.cfg.policy.policy_type == 'BCDPPolicy':
            repeated_diffusion_steps = self.cfg.policy.policy_head.network_kwargs.repeated_diffusion_steps
            actions = data["actions"]
            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            dist = dist.mean(dim=1, keepdim=True)
            features_repeated = dist.repeat(repeated_diffusion_steps, 1, 1)
            bc_loss = self.model.policy_head.loss(actions_repeated, features_repeated) 
        else:
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"], reduction="mean")
        mi_loss = self.mine.get_mi(x, z) * self.cfg.train.mi_loss_scale

        return bc_loss, mi_loss
    
    def compute_mine_model_loss(self, data):
        with torch.no_grad():
            x, z, _ = self.model(data, return_latent=True)
        x, z = x.detach(), z.detach()
        mi_loss = self.mine(x, z) * self.cfg.train.mine_mi_loss_scale

        return mi_loss
        
    @torch.no_grad()
    def evaluate(self, tag="val"):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0
        self.model.eval()

        for data in tqdm(self.val_loader):
            data = self.model.preprocess_input(data, augmentation=False)
            bc_loss, mi_loss, mmd, kl_div = self.compute_loss(data)
            loss = bc_loss + mi_loss

            ret_dict = {
                "loss": loss.item(),
                "bc_loss": bc_loss.item(),
                "mi_loss": mi_loss.item(),
                "mine": -mi_loss.item(),
            }

            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if cfg.train.debug:
                break

        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        return out_dict
