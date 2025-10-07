"""
A Clean A1 Arm Deployment Script
Author: Wenkai
"""

import time
import numpy as np
import torch
import requests
import json
import os
from PIL import Image
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

def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class A1ArmDeploy:
    
    def __init__(self, controller, model_type="dp", 
                 websocket_host="0.0.0.0", websocket_port=8000,
                 cogact_url='http://127.0.0.1:5500/api/inference',
                 model_path = "/home/luka/Wenkai/openpi/examples/a1_deploy/rgb.ckpt" # pp_rgb.ckpt scube_rgbd
                 ):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.controller = controller
        self.model_type = model_type.lower()
        self.pipelines = None
        
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

    def _init_model_client(self, model_path):
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
        color_image = color_image.resize((256, 256))
        depth_frame = depth_frame.resize((256, 256))

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
        image = image.resize((256, 256))

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

            return bgr_main, bgr_wrist, main_depth
            
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
            return True
        else:
            print(f"Preset {preset_name} not found")
            return False
            
    def element_retrieve(self, main_image, wrist_image, main_depth, current_state, last_main_image, last_wrist_image, last_main_depth, last_status, prompt=None):
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


def main():
    from controller_eef import A1ArmController
    
    # contoller and robot system init
    controller = A1ArmController()
    robot_system = A1ArmDeploy(controller, model_type="dp") # cogact or pi0 or memo-vla or openvla
    
    robot_system.run_control_loop(
        n_iterations=1000,  # 总迭代次数
        chunk_size=8,  # 推理后保留的步数
        merge_step=1,      # 合并的步数, 1 for CogACT ONLY
        loop_interval=0.1  # 控制循环间隔
    )


if __name__ == "__main__":
    main()
