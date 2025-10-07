import torch
import torch.nn as nn
import robomimic.utils.tensor_utils as TensorUtils

from .modules.data_augmentation import (
    IdentityAug,
    TranslationAug,
    ImgColorJitterAug,
    ImgColorJitterGroupAug,
    BatchWiseImgColorJitterAug,
    DataAugGroup,
)

REGISTERED_POLICIES = {}


def register_policy(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_POLICIES:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_POLICIES[policy_name] = policy_class


def get_policy_class(policy_name):
    """Get the policy class from the registry."""
    if policy_name.lower() not in REGISTERED_POLICIES:
        raise ValueError(
            "Policy class with name {} not found in registry".format(policy_name)
        )
    return REGISTERED_POLICIES[policy_name.lower()]


def get_policy_list():
    return REGISTERED_POLICIES


class PolicyMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all policies that should not be registered here.
        _unregistered_policies = ["BasePolicy"]

        if cls.__name__ not in _unregistered_policies:
            register_policy(cls)
        return cls


class BasePolicy(nn.Module, metaclass=PolicyMeta):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.train.device
        self.shape_meta = shape_meta

        policy_cfg = cfg.policy

        # add data augmentation for rgb inputs
        color_aug = eval(policy_cfg.color_aug.network)(
            **policy_cfg.color_aug.network_kwargs
        )

        policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta[
            "all_shapes"
        ][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(policy_cfg.translation_aug.network)(
            **policy_cfg.translation_aug.network_kwargs
        )
        self.img_aug = DataAugGroup((color_aug, translation_aug))

    def forward(self, data):
        """
        The forward function for training.
        """
        raise NotImplementedError

    def get_action(self, data):
        """
        The api to get policy's action.
        """
        raise NotImplementedError

    def _get_img_tuple(self, data):
        img_tuple = tuple(
            [data["obs"][img_name] for img_name in self.image_encoders.keys()]
        )
        return img_tuple

    def _get_aug_output_dict(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(self.image_encoders.keys())
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True, augmentation=None):
        if train_mode:  # apply augmentation
            if augmentation == None:
                augmentation = self.cfg.train.use_augmentation
            if augmentation:
                img_tuple = self._get_img_tuple(data)
                aug_out = self._get_aug_output_dict(self.img_aug(img_tuple))
                for img_name in self.image_encoders.keys():
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension. if need, .to(self.device)
            )
            data["task_emb"] = data["task_emb"].squeeze(1)  #.to(self.device)
        return data

    def compute_loss(self, data, reduction="mean", augmentation=None):
        data = self.preprocess_input(data, train_mode=True, augmentation=augmentation)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        return loss

    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        pass

    def save(self, path, epoch, optimizer, scheduler):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path, optimizer=None, scheduler=None, map_location='cpu'):
        checkpoint = torch.load(path, map_location=map_location)
        epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return epoch
