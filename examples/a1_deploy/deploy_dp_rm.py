"""
A Clean A1 Arm Deployment Script
Author: Wenkai
"""

import time, signal
import numpy as np
import torch
import requests
import json
import os
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
# from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
import pyrealsense2 as rs
# from openpi_client import websocket_client_policy
# from transformers import AutoModelForVision2Seq, AutoProcessor
from memory import TrajMatcher, infer_action

import dill
import hydra
import cv2
# from diffusion_policy.real_world.real_env import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
# from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from controller_eef import A1ArmController
import controller_eef

# --- robomimic imports ---
from robomimic.utils.file_utils import policy_from_checkpoint, maybe_dict_from_checkpoint, config_from_checkpoint
#from robomimic.algo import RolloutPolicy   # returned by policy_from_checkpoint internally
#from robomimic.config import config_factory
#from robomimic.algo.algo import algo_factory, RolloutPolicy
from robomimic.utils.obs_utils import initialize_obs_utils_with_config
#from robomimic.utils.action_utils import vector_to_action_dict, action_dict_to_vector

def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

# ----------- helpers --------------
def _sigint(_sig, _frm):
    global running
    running = False          

def _pick_stats(stats: dict, action_key='actions'):
    """Pick the right stats dict for the action key."""
    if action_key in stats:
        return stats[action_key]
    # fall back to the only entry if a custom key was used
    return next(iter(stats.values()))

def load_action_min_max(ckpt_path, action_key='actions'):
    """
    Read a_min/a_max for deployment from a robomimic v0.5 checkpoint.

    Supports either {min,max} or {offset,scale} formats.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    stats = (
        ckpt.get('action_normalization_stats')
        or ckpt.get('rollout', {}).get('action_normalization_stats')
        or ckpt.get('metadata', {}).get('action_normalization_stats')
    )
    if stats is None:
        raise KeyError("action_normalization_stats not found in checkpoint")

    s = _pick_stats(stats, action_key)

    if 'min' in s and 'max' in s:
        a_min, a_max = np.asarray(s['min']), np.asarray(s['max'])
    else:
        # robomimic uses offset/scale; inverse is x = x_norm / scale - offset
        scale, offset = np.asarray(s['scale']), np.asarray(s['offset'])
        a_min = (-1.0 / scale) - offset
        a_max = ( 1.0 / scale) - offset
    return a_min, a_max

def denorm_action_(a_hat, a_min, a_max):
    """
    Inverse of min-max to [-1,1]:
        a = a_min + (a_hat + 1)/2 * (a_max - a_min)
    Works per-dimension on vectors.
    """
    return a_min + 0.5 * (a_hat + 1.0) * (a_max - a_min)

# act_norm: np.ndarray shape (A,) or (B, A) from: act = policy(ob=obs)
# ckpt_dict: from FileUtils.policy_from_checkpoint(...)
def denorm_action(act_norm, ckpt_dict, fallback_min=None, fallback_max=None):
    stats = ckpt_dict.get("action_normalization_stats", None)
    if stats is None:
        # Fallback for when stats weren't saved: invert min_max if you know bounds.
        if fallback_min is None or fallback_max is None:
            raise ValueError("No action_normalization_stats in ckpt and no fallback bounds provided.")
        # min_max to [-1,1] forward: z = 2*(x - min)/(max - min) - 1
        # inverse:
        return (act_norm + 1.0) * 0.5 * (fallback_max - fallback_min) + fallback_min

    # Common case: a single flattened action key (e.g., "actions")
    if "actions" in stats:
        offset = stats["actions"]["offset"]  # shape (1, A)
        scale  = stats["actions"]["scale"]   # shape (1, A)
        return act_norm * scale + offset

    # Multi-key case: stitch per-key segments in the order used at train time
    keys = ckpt_dict["config"]["train"]["action_keys"]  # preserved by robomimic
    out = []
    start = 0
    for k in keys:
        d = stats[k]["offset"].shape[1]
        seg = act_norm[..., start:start+d]
        out.append(seg * stats[k]["scale"] + stats[k]["offset"])
        start += d
    return np.concatenate(out, axis=-1)

class A1ArmDeploy:
    
    def __init__(self, controller, model_type="dp",
                 websocket_host="0.0.0.0", websocket_port=8000,
                 cogact_url='http://127.0.0.1:5500/api/inference',
                 model_path = "/home/luka/Wenkai/openpi/examples/a1_deploy/bls1.pth" # pp_rgb.ckpt scube_rgbd
                 ):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.controller = controller
        self.model_type = model_type.lower()
        self.pipelines = None
        self.policy = None
        self.rm_cfg = None
        self.ckpt_dict = None
        self.shape_meta = None
        self.expected_obs_keys = None   # filled after load
        
        # 常用位置预设
        self.position_presets = {
            "pick_init": (0.283, 0.049, 0.15, 0.00, 0.00, 1, 0, 1),
            "stack_init": (0.243, 0.069, 0.17, 0.00, 0.00, 1, 0, 1),
            "press_init": (0.253, 0.15, 0.18, 0.00, 0.00, 1, 0, 1),
            "swap_init": (0.314, 0.15, 0.193, 0.00, 0.00, 1, 0, 1),
            "swap_init_var1": (0.253, 0.092, 0.190, 0.00, 0.00, 1, 0, 1),
            "swap_init_var2": (0.23, 0.09, 0.164, 0.00, 0.00, 1, 0, 1),
            "swap_init_var3": (0.442, 0.183, 0.174, 0.00, 0.00, 1, 0, 1),

            "env_show": (0.360, 0.1, 0.35, 0.00, 0.00, 1, 0, 1),
            "pick_bowl": (0.320, 0.12, 0.2456, 0.00, 0.00, 1, 0, 1), # demo set
            "stack_cup": (0.4, 0.26, 0.28, 0.00, 0.00, 1, 0, 1),
            "pick_plate": (0.394, 0.182, 0.257, 0.00, 0.00, 1, 0, 1),
            
            "pick_bowl_mean": (0.320,  0.061,  0.2456, 0.00, 0.00, 1, 0, 1),
            "stack_cup_mean":(0.4, 0.22,  0.30, 0.00, 0.00, 1, 0, 1),  # (0.3862, 0.296, 0.3333, -0.0958, 0.9948, -0.0185, -0.0304, 1) # (0.4, 0.26,  0.28, 0.00, 0.00, 1, 0, 1)
            "pick_plate_mean":(0.394,  0.182,  0.257, 0.00, 0.00, 1, 0, 1),
            
            # TODO
            "stack_two_cups":(0.236,  0.251,  0.30, 0.00, 0.00, 1.00, 0.00, 1.00),
            # "stack_two_cups":(0.23,  0.31,  0.3, 0.00, 0.00, 1.00, 0.00, 1.00),
        }
        
        self.pos_limits = {
            "x_min": 0.1, "x_max": 0.60,  # 0.35/0.45 for stack cup, 0.23/0.41 for swap, 0.2/0.40 for demo pick bowl
            "y_min": 0.1, "y_max": 0.60, # 0.34 for runhao, 0.27 for CogACT,  -0.03/0.35 for stack cup, -0.02/0.32 for demo pick bowl
            "z_min": 0.03, "z_max": 0.50 # 0.11/0.35 for runhao, 0.1/0.27 for CogACT, 0.13/0.35 for cup
        }
        
        # no need for dp
        # self.default_prompt = "place the green cube and orange into the bowl"
        
        self.data = {
        'task_description': "swap the slots of two objects"
        }

        self._init_model_client(model_path)

    # ---------- REPLACE your _init_model_client ----------
    def _init_model_client(self, model_path: str):

        device = self.device

        # --- load checkpoint dict (v0.5: just the path) ---
        self.policy, self.ckpt = policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=True)

        '''
        # --- resolve algo_name (allow manual override) ---
        algo_in_ckpt = ckpt.get("algo_name") if isinstance(ckpt, dict) else None
        if algo_in_ckpt is None and isinstance(ckpt, dict):
            cfg_blob = ckpt.get("config") or ckpt.get("train_config") or ckpt.get("rm_cfg")
            if isinstance(cfg_blob, dict):
                algo_in_ckpt = cfg_blob.get("algo_name") or (cfg_blob.get("algo", {}) or {}).get("name")
        final_algo_name = algo_name or algo_in_ckpt
        if final_algo_name is None:
            raise RuntimeError("Checkpoint lacks 'algo_name'. Pass algo_name='bc'|'bc_rnn'|'diffusion_policy'.")

        # --- rebuild config (prefer exact one from ckpt) ---
        try:
            rm_cfg, _ = config_from_checkpoint(algo_name=final_algo_name, ckpt_dict=ckpt)
        except Exception:
            rm_cfg = config_factory(algo_name=final_algo_name)

        # --- force eval-only behavior: disable schedulers & co. ---
        def sanitize_rm_cfg_for_eval(cfg):
            # unlock just in case the config is locked
            try:
                ctx = cfg.value_unlocked()  # upstream supports value-level unlocking
            except Exception:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                # ensure tiny numeric train loop defaults in case your fork reads them
                if "train" in cfg and hasattr(cfg.train, "num_epochs"):
                    try:
                        cfg.train.num_epochs = int(cfg.train.num_epochs)
                    except Exception:
                        cfg.train.num_epochs = 1
                if "experiment" in cfg and hasattr(cfg.experiment, "gradient_steps_per_epoch"):
                    try:
                        cfg.experiment.gradient_steps_per_epoch = int(cfg.experiment.gradient_steps_per_epoch)
                    except Exception:
                        cfg.experiment.gradient_steps_per_epoch = 1

                # disable LR schedulers for ALL nets under algo.optim_params
                if "algo" in cfg and "optim_params" in cfg.algo:
                    for net_name, net_cfg in list(cfg.algo.optim_params.items()):
                        lr = getattr(net_cfg, "learning_rate", None)
                        if lr is not None:
                            # upstream JSON shows keys like {"scheduler_type": "multistep", "epoch_schedule": [...]}
                            lr.scheduler_type = "none"
                            if hasattr(lr, "epoch_schedule"):
                                lr.epoch_schedule = []

        #sanitize_rm_cfg_for_eval(rm_cfg)

        # --- shapes (from ckpt if present, else user-provided / defaults) ---
        shape_meta = ckpt.get("shape_metadata") if isinstance(ckpt, dict) else None
        if shape_meta is None:
            if obs_key_shapes is None:
                # sensible A1 defaults; adjust to your cameras / eef vector
                obs_key_shapes = {
                    "main_camera": (3, 84, 84),
                    "wrist_camera": (3, 84, 84),
                    "robot_eef_pose": (7,),
                }
            if ac_dim is None:
                ac_dim = 7
            shape_meta = {"all_shapes": obs_key_shapes, "ac_dim": ac_dim}

        # --- build Algo and load weights (official test-time pattern) ---
        model = algo_factory(
            final_algo_name,
            rm_cfg,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )

        # find a state_dict
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(k, str) and "." in k for k in ckpt.keys()):
            state_dict = ckpt  # raw state_dict at top-level
        else:
            state_dict = ckpt   # assume raw state_dict

        model.deserialize(state_dict)
        model.set_eval()

        # rollout wrapper (one action per call)
        self.policy = RolloutPolicy(model)
        self.ckpt_dict = ckpt if isinstance(ckpt, dict) else {"model": "state_dict_only"}
        self.rm_cfg = rm_cfg
        self.shape_meta = shape_meta
        self.model_type = "robomimic"

        print(f"[robomimic] Loaded algo={final_algo_name}, ac_dim={shape_meta['ac_dim']}, "
            f"obs_keys={list(shape_meta['all_shapes'].keys())}")
        '''

    def _init_model_client_old(self, model_path):
        if self.model_type == 'dp':
            payload = torch.load(open(model_path, 'rb'), pickle_module=dill)
            self.cfg = payload['cfg']
            cls = hydra.utils.get_class(self.cfg._target_)
            workspace = cls(self.cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # hacks for method-specific setup.
            action_offset = 0
            delta_action = False
            if 'diffusion' in self.cfg.name:
                # diffusion model
                self.policy: BaseImagePolicy
                self.policy = workspace.model
                if self.cfg.training.use_ema:
                    self.policy = workspace.ema_model

                self.policy.eval().to(self.device)

                # set inference params
                self.policy.num_inference_steps = 16 # DDIM inference iterations
                self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
            else:
                raise RuntimeError("Unsupported policy type: ", self.cfg.name)

            # TODO magic numbers
            frequency = 10
            steps_per_inference = 6
            dt = 1/frequency

            obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
            n_obs_steps = self.cfg.n_obs_steps
            print("n_obs_steps: ", n_obs_steps)
            print("steps_per_inference:", steps_per_inference)
            print("action_offset:", action_offset)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Choose 'pi0' or 'cogact' or 'openvla'.")
    
    def initialize_camera(self):
        context = rs.context()
        connected_devices = [device.get_info(rs.camera_info.serial_number) for device in context.devices]
        print("Connected camera serials:", connected_devices)
        
        #'L515': 'f0265239', # 'f0265239', f0210138    # 替换为 L515 的序列号 f0265239 f0210138
        #'D435i': '332522071841',
            
        self.d435i_serial = '332522071841'
        self.l515_serial = 'f0210138'
        
        # 映射相机型号到序列号（请替换为您的相机实际序列号）
        try:
            print("Initializing D435i...")
            self.pipeline_d435i = rs.pipeline()
            config_d435i = rs.config()
            config_d435i.enable_device(self.d435i_serial)
            config_d435i.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline_d435i.start(config_d435i)
            time.sleep(0.5)
            
            # 初始化L515 - 同时启用彩色和深度流
            print("Initializing L515...")
            self.pipeline_l515 = rs.pipeline()
            config_l515 = rs.config()
            config_l515.enable_device(self.l515_serial)
            
            # 同时启用彩色和深度流
            config_l515.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config_l515.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

            # 启动pipeline
            self.profile_l515 = self.pipeline_l515.start(config_l515)
            
            # 创建对齐对象，将深度对齐到彩色
            self.align = rs.align(rs.stream.color)
            time.sleep(1.0)
            
            print("Both cameras initialized successfully")
            self._warm_up_camera()
            
        except Exception as e:
            print("Failed to setup cameras: %s", str(e))
            raise

        self._warm_up_camera()

    def get_L515_image(self):
        """
        从 pipelines 中获取 L515 相机的图像，转换为 PIL Image 并调整为 (384, 384) 大小。
        """
        frames_l515 = self.pipeline_l515.wait_for_frames()
        aligned_frames = self.align.process(frames_l515)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()        
        if not color_frame:
            print("No color frame captured from L515 camera.")
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        color_image = Image.fromarray(color_image)
        depth_frame = np.asanyarray(depth_frame.get_data())
        depth_frame = depth_frame / 10000.0 
        depth_frame = np.clip(depth_frame, 0, 1)  # 裁
        # depth_colormap = cv2.applyColorMap(
        #     cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
        # )
        depth_frame = (depth_frame * 255).astype(np.uint8)
        depth_frame = Image.fromarray(depth_frame)
        color_image = ImageOps.fit(color_image, (84, 84), method=Image.BILINEAR, centering=(0.5, 0.5))
        depth_frame = ImageOps.fit(depth_frame, (84, 84), method=Image.NEAREST,  centering=(0.5, 0.5))

        return color_image, depth_frame

    def get_D435_image(self):
        """
        从 pipelines 中获取 D435 相机的图像，转换为 PIL Image 并调整为 (384, 384) 大小。
        """
        frames = self.pipeline_d435i.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("No color frame captured from D435i camera.")
            return None
        color_image = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color_image)
        image = ImageOps.fit(image, (84, 84), method=Image.NEAREST,  centering=(0.5, 0.5))
        return image

    def _warm_up_camera(self, frames=10):
        for _ in range(frames):
            self.get_L515_image()
            self.get_D435_image()
        
    def capture_images(self, step=None):
        try:
            main_image, main_depth = self.get_L515_image()
            wrist_image = self.get_D435_image()
            
            if main_image is None or wrist_image is None:
                print("Failed to capture image from camera.")
                return None, None, None
                
            bgr_main = self._convert_to_bgr(main_image)
            bgr_wrist = self._convert_to_bgr(wrist_image)
            
            if step is not None:
                save_dir = "/home/luka/Wenkai/visualization/"
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"captured_image_{step}.png")
                wrist_path = os.path.join(save_dir, f"captured_image_wrist_{step}.png")
                depth_path = os.path.join(save_dir, f"captured_image_depth_{step}.png")
                bgr_main.save(main_path)
                bgr_wrist.save(wrist_path)
                main_depth.save(depth_path)

            #return bgr_main, bgr_wrist, main_depth
            return main_image, wrist_image, main_depth
            
        except Exception as e:
            print(f"Failed to get image from camera: {e}")
            return None, None
    
    def _convert_to_bgr(self, image):
        """将图像转换为BGR格式"""
        rgb_image = image.convert("RGB")
        channels = rgb_image.split()[::-1]  
        return Image.merge("RGB", channels)
        
    def quaternion_to_rpy(self, quaternion):
        r = R.from_quat(quaternion)
        return r.as_euler('xyz', degrees=False)
        
    def preset(self, preset_name):
        if preset_name in self.position_presets:
            self.controller.execute_eef(self.position_presets[preset_name], preset_name)
            print(self.position_presets[preset_name])
            return True
        else:
            print(f"Preset {preset_name} not found")
            return False
            
    # ---------- REPLACE your element_retrieve ----------
    def element_retrieve(self,
        main_image, wrist_image, current_state,
        last_main_image, last_wrist_image, last_status
    ):
        #print(main_image)
        
        return {
            "obs_rgb": np.stack([last_main_image,  main_image],  axis=0).astype(np.uint8),       # (T, H, W, 3) uint8
            "wrist_image_rgb": np.stack([last_wrist_image, wrist_image], axis=0).astype(np.uint8),
            #"eef_states": np.stack([last_status, current_state], axis=0).astype(np.float32),       # (T, 7)
        }
        
        '''
        def last_frame(x):
            return x[-1] if (isinstance(x, np.ndarray) and x.ndim >= 3 and x.shape[0] > 4) else x

        def to_uint8_img(x):
            x = np.asarray(x.convert("RGB")) if isinstance(x, Image.Image) else np.asarray(x)
            if x.dtype != np.uint8:
                x = (x * (255 if x.max() <= 1 else 1)).clip(0, 255).astype(np.uint8)
            return x

        obs = {}

        # Map your sensor names to the keys the checkpoint expects.
        # Adjust these names if your checkpoint used different ones.
        # Common examples seen in robosuite runs:
        #   "main_camera" or "agentview_image"
        #   "wrist_camera" or "robot0_eye_in_hand_image"
        #   "depth_camera" or "agentview_depth"
        #   "robot_eef_pose" (low-dim)
                
        T = int(self.rm_cfg.algo.horizon.observation_horizon)  # e.g., 2
        obs = {}
        rgb_keys = list(self.rm_cfg.observation.modalities.obs.rgb)
        low_keys = list(self.rm_cfg.observation.modalities.obs.low_dim)
        imgs = [main_image, wrist_image]  # map your sensors in order

        for i, k in enumerate(rgb_keys):
            C, H, W = self.shape_meta["all_shapes"][k]
            img = imgs[i] if i < len(imgs) else Image.new("RGB", (W, H))
            img = np.asarray(img.convert("RGB").resize((W, H)), np.uint8)  # HWC uint8
            obs[k] = np.repeat(img[None, ...], T, axis=0)                  # T,H,W,3

        for k in low_keys:
            D = self.shape_meta["all_shapes"][k][0]
            v = np.asarray(current_state, np.float32)
            v = v if v.shape[0] == D else np.zeros((D,), np.float32)
            obs[k] = np.repeat(v[None, ...], T, axis=0)                    # T,D

        # If your checkpoint used different key names, remap here based on self.expected_obs_keys
        # (Optional) sanity check:
        if self.expected_obs_keys is not None:
            missing = [k for k in self.expected_obs_keys if k not in obs]
            if len(missing) > 0:
                print("[robomimic] WARNING missing obs keys for this ckpt:", missing)

        return obs
        '''
    
    def element_retrieve_old(self, main_image, wrist_image, main_depth, current_state, last_main_image, last_wrist_image, last_main_depth, last_status, prompt=None):
        # pi prepare
        obs_dict_np = dict()
        obs_shape_meta = self.cfg.task.shape_meta['obs']
        
        img_array = np.asarray(main_image).astype(np.uint8)[None, ...]
        img_wrist_array = np.asarray(wrist_image).astype(np.uint8)[None, ...]
        img_depth_array = np.asarray(main_depth).astype(np.uint8)[None, ...]
        img_depth_array = np.repeat(img_depth_array[..., np.newaxis], 3, axis=-1)  # T H W C
        eef_pose = current_state[:3][None, ...]
        
        if last_main_image is None:
            img_array = np.repeat(img_array, 2, axis=0)  # T H W C
            img_wrist_array = np.repeat(img_wrist_array, 2, axis=0)  # T H W C
            img_depth_array = np.repeat(img_depth_array, 2, axis=0)  # T H W C
            eef_pose = np.repeat(eef_pose, 2, axis=0)  # T H W C
        else:
            last_main_image = np.asarray(last_main_image).astype(np.uint8)[None, ...]
            last_wrist_image = np.asarray(last_wrist_image).astype(np.uint8)[None, ...]
            last_main_depth = np.asarray(last_main_depth).astype(np.uint8)[None, ...]
            last_main_depth = np.repeat(last_main_depth[..., np.newaxis], 3, axis=-1)  # T H W C
            last_status = last_status[:3][None, ...]
            img_array = np.concatenate((last_main_image, img_array), axis=0)
            img_wrist_array = np.concatenate((last_wrist_image, img_wrist_array), axis=0)
            img_depth_array = np.concatenate((last_main_depth, img_depth_array), axis=0)
            eef_pose = np.concatenate((last_status, eef_pose), axis=0)

        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            shape = attr.get('shape')
            if type == 'rgb':
                if key == 'main_camera':
                    this_imgs_in = img_array
                elif key == 'wrist_camera':
                    this_imgs_in = img_wrist_array
                elif key == 'depth_camera':
                    this_imgs_in = img_depth_array
                t,hi,wi,ci = this_imgs_in.shape
                co,ho,wo = shape
                assert ci == co
                out_imgs = this_imgs_in
                if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                    tf = get_image_transform(
                        input_res=(wi,hi), 
                        output_res=(wo,ho), 
                        bgr_to_rgb=False)
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                    if this_imgs_in.dtype == np.uint8:
                        out_imgs = out_imgs.astype(np.float32) / 255
                # THWC to TCHW
                obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
            elif type == 'low_dim':
                if key == 'robot_eef_pose':
                    this_data_in = eef_pose
                else:
                    raise NotImplementedError
                obs_dict_np[key] = this_data_in

        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

        return obs_dict

    def process_action(self, action_step, current_position, current_rpy):
        position = action_step[0:3]
        
        position[0] = min(max(position[0], self.pos_limits["x_min"]),
                                self.pos_limits["x_max"])
        position[1] = min(max(position[1], self.pos_limits["y_min"]), 
                                self.pos_limits["y_max"])
        position[2] = min(max(position[2], self.pos_limits["z_min"]), 
                                self.pos_limits["z_max"])
        
        # rpy = action_step[3:6]
        rpy = np.array([0.027, -0.01, 3.15])  # 固定姿态，可修改为 current_rpy + delta_rpy
        
        gripper_position = action_step[6]
        print(gripper_position)
        if gripper_position > 0.5:
            gripper_position = 1
        else:
            gripper_position = 0
        print(gripper_position)
        
        return position, rpy, gripper_position

    def run_control_loop(self, n_iterations=200, chunk_size=4, merge_step=2, loop_interval=0.1):
        """
        Control loop compatible with robomimic .pth checkpoints (step-wise inference).
        - Expects self.policy to be a robomimic RolloutPolicy (from policy_from_checkpoint)
        - Uses self.element_retrieve(...) to build a single-step obs dict
        - Calls self.controller.execute_eef(...) with [pos(3), rpy(3), grip(1)]
        Notes:
        * robomimic policies return ONE action per call; no horizon/sequence merging.
        * If your policy returns an action dict, we flatten it in the order rm_cfg.train.action_keys.
        """
        # --- Sensors / task init ---
        self.initialize_camera()

        task_name = "stack_two_cups"
        self.preset(task_name)  # preset the initial position from the task
        time.sleep(3.0)

        self.action_history = []

        execute_action = self.position_presets[task_name]
        init_pose = execute_action[0:3]
        init_quat = execute_action[3:7]
        init_rpy = self.quaternion_to_rpy(init_quat)
        init_gripper = -execute_action[-1]

        print("Initial RPY:", init_rpy)

        step = 0
        last_main_image, last_wrist_image, last_main_depth = None, None, None
        last_status = None

        # --- robomimic policies: start a fresh episode ONCE (not every step) ---
        if hasattr(self.policy, "start_episode"):
            self.policy.start_episode()  # prepares normalization, history, eval()
        else:
            print("[warn] policy has no start_episode(); proceeding without it.")

        
        global_start_time = time.time()
        time_run = time.time() - global_start_time
        while step < n_iterations and time_run < 10.0:
            time_run = time.time() - global_start_time
            try:
                print(f"\n--- Step {step} ---")
                start_time = time.time()

                # Acquire images
                main_image, wrist_image, main_depth = self.capture_images(step)
                if main_image is None or wrist_image is None:
                    # keep loop cadence even if cameras hiccup
                    elapsed_s = time.time() - start_time
                    sleep_time = loop_interval - elapsed_s
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                
                if last_main_image is None or last_wrist_image is None or last_main_depth is None:
                    last_main_image, last_wrist_image, last_main_depth = main_image, wrist_image, main_depth
                    
                # Build current low-dim state vector to feed in obs (eef pose etc.)
                current_state = np.concatenate((init_pose, init_rpy, [init_gripper]), axis=0)
                if last_status is None:
                    last_status = current_state
                
                obs_dict = self.element_retrieve(
                    main_image=main_image, 
                    wrist_image=wrist_image, 
                    #main_depth=None, 
                    current_state=current_state, 
                    last_main_image=last_main_image, 
                    last_wrist_image=last_wrist_image, 
                    #last_main_depth=None, 
                    last_status=last_status
                )
                
                inf_time = time.time()
                action = self.policy(ob=obs_dict)
                new_pos, new_rpy, grip = self.process_action(action, init_pose, init_rpy)
                init_pose, init_rpy, init_gripper = new_pos, new_rpy, grip
                final_action = np.concatenate((new_pos, new_rpy, [grip]), axis=0)
                print(f"Executing: {final_action}")
                # import pdb;pdb.set_trace()
                self.controller.execute_eef(final_action, task_name)
                
                '''
                n_steps = min(len(all_actions), chunk_size)
                for step_idx in range(0, n_steps, merge_step): 
                    last_main_image, last_wrist_image, last_main_depth = self.capture_images(step+step_idx)
                    last_status = init_pose
                    
                    merged_chunk = all_actions[step_idx : step_idx + merge_step]
                    merged_action_prefix = np.sum(merged_chunk[:, 0:6], axis=0)
                    gripper_command = merged_chunk[-1][6]
                    action_step = np.concatenate([merged_action_prefix, [gripper_command]])

                    new_pos, new_rpy, grip = self.process_action(action_step, init_pose, init_rpy)

                    init_pose, init_rpy, init_gripper = new_pos, new_rpy, grip
                    final_action = np.concatenate((new_pos, new_rpy, [grip]), axis=0)
                    print(f"Executing: {final_action}")
                    # import pdb;pdb.set_trace()

                    self.controller.execute_eef(final_action, task_name)
                '''
                
                '''''
                # fill in last step
                if last_main_image is None or last_wrist_image is None or last_main_depth is None:
                    last_main_image, last_wrist_image, last_main_depth = main_image, wrist_image, main_depth
                if last_status is None:
                    last_status = current_state

                # --- Build single-step observation dict (HWC uint8 images; 1D float low-dim) ---
                obs_dict = self.element_retrieve(
                    main_image, wrist_image, current_state,
                    last_main_image, last_wrist_image, last_status
                )

                # --- Inference ---
                inf_time = time.time()

                # robomimic RolloutPolicy -> one action per call
                action = self.policy(obs_dict)  # np.ndarray or dict
                
                # denormalize actions
                #action = denorm_action(action_norm, self.ckpt)
                action[-1] = 1.0 if action[-1] > 0 else -1.0                   # optional

                all_actions = action[None, :]                                   # (1, ac_dim)
                                        
                print(f"Inference time: {time.time() - inf_time:.4f}s")
                
                # --- Execute action(s) ---
                # Single step: directly apply one action vector
                action_step = all_actions[0]
                new_pos, new_rpy, grip = self.process_action(action_step, init_pose, init_rpy)
                init_pose, init_rpy, init_gripper = new_pos, new_rpy, grip
                final_action = np.concatenate((new_pos, new_rpy, [grip]), axis=0)
                print(f"Executing: {final_action}")
                self.controller.execute_eef(final_action, task_name)
                '''''

                # update any “last_*” placeholders if your element_retrieve uses them
                last_main_image, last_wrist_image, last_main_depth = main_image, wrist_image, main_depth
                last_status = init_pose


                # --- Timing (fixed the units: use seconds for sleep calculation) ---
                elapsed_s = time.time() - start_time
                print(f"Total step time: {elapsed_s * 1000.0:.2f} ms")

                sleep_time = loop_interval - elapsed_s
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Control loop is running slower than desired frequency")

                step += 1
            
            except KeyboardInterrupt:
                print("\nControl loop interrupted by user")
            except Exception as e:
                import traceback
                print(f"An unexpected error occurred: {e}")
                traceback.print_exc()
            finally:
                print("Control loop ended")

        # --- Wrap-up: release & return to preset pose ---
        try:
            # open gripper (+1) while keeping last pose
            final_action = np.concatenate((init_pose, init_rpy, [1]), axis=0)
            self.controller.execute_eef(final_action, task_name)
            time.sleep(3.0)
        finally:
            time.sleep(0.5)
            self.controller.execute_eef(execute_action, task_name)
            time.sleep(3.0)

    def test_loop(self, itr=10):
        
        self.initialize_camera()

        task_name = "stack_two_cups"
        print("---------------- presetting -------------------------------------")
        self.preset(task_name)  # preset the initial position from the task
        print("---------------- preset complete -------------------------------------")
        #time.sleep(100)
        
        pos = [0.236,  0.251,  0.30, 0.00, 0.00, 1.00, 0.00, 1.00]
        for i in range(itr):
            self.controller.execute_eef(pos, "test")
        

        
    def run_control_loop_old(self, n_iterations=200, chunk_size=4, merge_step=2, loop_interval=0.1):         
        self.initialize_camera()
        
        task_name = "stack_two_cups"
        self.preset(task_name) #  preset the initial position from the task
        import pdb;pdb.set_trace()
        
        self.action_history = []
        # "swap_init", "pick_init", "stack_init", "press_init"
        # "pick_bowl", "stack_cup", "pick_plate"
        # "pick_bowl_mean", "stack_cup_mean", "pick_plate_mean"
        # to show the environment: "env_show"
                
        execute_action = self.position_presets[task_name]
        init_pose = execute_action[0:3]
        init_quat = execute_action[3:7]
        init_rpy = self.quaternion_to_rpy(init_quat)
        init_gripper = -execute_action[-1]
        
        print("Initial RPY:", init_rpy)
        
        step = 0
        states = []
        last_main_image, last_wrist_image, last_main_depth = None, None, None
        last_status = None
        try:
            while step < n_iterations:
                print(f"\n--- Step {step} ---")
                start_time = time.time()

                main_image, wrist_image, main_depth = self.capture_images(step)
                if main_image is None or wrist_image is None:
                    time.sleep(loop_interval)
                    continue
                
                
                current_state = np.concatenate((init_pose, init_rpy, [init_gripper]), axis=0)
                
                if self.model_type == "dp":
                    self.policy.reset()
                    obs_dict = self.element_retrieve(main_image, wrist_image, main_depth, current_state, last_main_image, last_wrist_image, last_main_depth, last_status)
                    inf_time = time.time()
                    result = self.policy.predict_action(obs_dict)
                    all_actions = result['action'][0].detach().to('cpu').numpy()
                    # loc = all_actions[3,:3]
                    # all_actions[:3, :3] = loc
                    # all_actions = all_actions[3:]

                # print(f"Action: {action}")
                print(f"Inference time: {time.time() - inf_time:.4f}s")

                n_steps = min(len(all_actions), chunk_size)
                for step_idx in range(0, n_steps, merge_step): 
                    last_main_image, last_wrist_image, last_main_depth = self.capture_images(step+step_idx)
                    last_status = init_pose
                    
                    merged_chunk = all_actions[step_idx : step_idx + merge_step]
                    merged_action_prefix = np.sum(merged_chunk[:, 0:6], axis=0)
                    gripper_command = merged_chunk[-1][6]
                    action_step = np.concatenate([merged_action_prefix, [gripper_command]])

                    new_pos, new_rpy, grip = self.process_action(action_step, init_pose, init_rpy)

                    init_pose, init_rpy, init_gripper = new_pos, new_rpy, grip
                    final_action = np.concatenate((new_pos, new_rpy, [grip]), axis=0)
                    print(f"Executing: {final_action}")
                    # import pdb;pdb.set_trace()

                    self.controller.execute_eef(final_action, task_name)
                
                # 计时并控制循环频率
                elapsed_time = (time.time() - start_time) * 1000  
                print(f"Total step time: {elapsed_time:.4f} ms")
                
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Control loop is running slower than desired frequency")
                    
                step += 1
                
            # 运行结束，回到初始位置
            final_action = all_actions[0][0:6] + [1]
            self.controller.execute_eef(final_action)
            time.sleep(1)
            self.controller.execute_eef(execute_action)
            
        except KeyboardInterrupt:
            print("\nControl loop interrupted by user")
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            print("Control loop ended")


class _A1ArmController(A1ArmController):
    def execute_eef(self, action, task):
        """
        :param action:
            - 7 params: [x, y, z, roll, pitch, yaw, gripper]  list[float]
            - 8 params: [x, y, z, w, x, y, z, gripper]  list[float]
        :return: success (bool)
        """
        try:
            if len(action) not in [7, 8]:
                controller_eef.rospy.logerr("action must be 7 or 8 elements")
                return False

            if len(action) == 7:
                print("action len is 7")
                r, p, y = action[3:6]
                quaternion = controller_eef.quaternion_from_euler(r, p, y)
                target_pos = action[:3]
                target_orientation = quaternion
                gripper_target = min(100, max(0, action[6] * 100))
            else:
                print("action len is 8")
                target_pos = action[:3]
                target_orientation = action[3:7]
                gripper_target = min(100, max(0, action[7] * 100))

            print("------------------- solve ik -----------------------------")
            joint_solution = self._ik_solver.solve_ik(
                target_position=target_pos, 
                target_orientation=target_orientation
            )
            if gripper_target < 80 and task == "pick_bowl":
                gripper_target = 30
            if gripper_target < 80 and task == "stack_cup" or "stack_two_cups":
                gripper_target = 0
            if gripper_target < 80 and task == "place_fruit_bowl":
                gripper_target = 50
            if gripper_target < 80 and task == "place_bread_plate":
                gripper_target = 10
            if gripper_target < 80 and task == "put_vege_bowl":
                gripper_target = 30
            print("------------------- gripper target gotten ----------------")
            target_joint = joint_solution.js_solution.position.cpu().numpy().flatten()
            print("------------------- step 1 done -------------------------")
            print("target joint:", target_joint, "gripper_target:", gripper_target)
            self._inter_and_pub_motion(target_joint, gripper_target)
            print("------------------- step 2 done -------------------------")

            return True
        
        except Exception as e:
            controller_eef.rospy.logerr(f"ERROR in execution: {e}")
            return False

    def _inter_and_pub_motion(self, target_joint, gripper_target, steps=1, hz=20):
    
        if len(target_joint) != 6:
            controller_eef.rospy.logerr("joint target must be 6 elements")
            return

        rate = controller_eef.rospy.Rate(hz)
        print("------------------- step 2.1 done -------------------------")
        
        joint_state = controller_eef.JointState()
        print("------------------- step 2.1.1 done -------------------------")
        joint_state.header.frame_id = 'world'
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        print("------------------- step 2.1.2 done -------------------------")
        joint_state.position = list(self._current_joint_position)
        print("------------------- step 2.1.3 done -------------------------")

        step_increment = [
            (target - current) / steps for target, current in zip(target_joint, joint_state.position)
        ]
        print("------------------- step 2.2 done -------------------------")

        start_time = time.time()
        for _ in range(steps):
            if controller_eef.rospy.is_shutdown():
                break
            joint_state.header.stamp = controller_eef.rospy.Time.now()
            joint_state.position = [
                current + increment for current, increment in zip(joint_state.position, step_increment)
            ]
            self._joint_pub.publish(joint_state)
            rate.sleep()
        # print(f"关节插值时间: {(time.time() - start_time) * 1000:.2f} ms")
        print("------------------- step 2.3 done -------------------------")

        gripper_start = self._current_gripper_position
        gripper_steps = 2
        gripper_increment = (gripper_target - gripper_start) / gripper_steps
        gripper_msg = controller_eef.gripper_position_control()
        gripper_msg.header = controller_eef.Header(frame_id="")
        print("------------------- step 2.4 done -------------------------")

        start_time = time.time()
        for i in range(gripper_steps + 1):
            if controller_eef.rospy.is_shutdown():
                break
            gripper_msg.header.stamp = controller_eef.rospy.Time.now()
            gripper_msg.gripper_stroke = gripper_start + gripper_increment * i
            self._gripper_pub.publish(gripper_msg)
            rate.sleep()
        # print(f"夹爪插值时间: {(time.time() - start_time) * 1000:.2f} ms")
        print("------------------- step 2.5 done -------------------------")
    
    
def main():
    
    
    #from controller_eef import A1ArmController
    
    # contoller and robot system init
    controller = _A1ArmController()
    robot_system = A1ArmDeploy(controller, model_type="rm") # cogact or pi0 or memo-vla or openvla
    #print(robot_system.get_D435_image())
    
    robot_system.test_loop()
    
    '''
    robot_system.run_control_loop(
        n_iterations=50,  # 总迭代次数
        chunk_size=8,  # 推理后保留的步数
        merge_step=1,      # 合并的步数, 1 for CogACT ONLY
        loop_interval=0.1  # 控制循环间隔
    )
    '''
    
    


if __name__ == "__main__":
    main()
