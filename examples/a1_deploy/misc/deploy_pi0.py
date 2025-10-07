# import huggingface_hub
# if not hasattr(huggingface_hub, "cached_download"):
#     from huggingface_hub import hf_hub_download
#     huggingface_hub.cached_download = hf_hub_download

import os
import json
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
import pyrealsense2 as rs
from pyrealsense_image import initialize_camera, stop_camera, get_L515_image, get_D435_image  
from openpi.examples.a1_deploy.misc.eef_control import execute_eef_IK
from eef_control_fast import A1ArmController
import time
# import re
# import dataclasses
# from openpi.policies import policy as _policy
# from openpi.policies import policy_config as _policy_config
# from openpi.training import config as _config
# import jax.numpy as jnp
from openpi_client import websocket_client_policy as _websocket_client_policy

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "../..")) 
# import hydra
# from hydra.core.global_hydra import GlobalHydra
# from third_party.world_models.mcts.world_model import WorldModel
# from third_party.world_models.mcts.utils import MCTS_RL

def get_device(cuda_number=0):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if cuda_number < device_count:
            print(f"Using cuda:{cuda_number} -> {torch.cuda.get_device_name(cuda_number)}")
            return torch.device(f"cuda:{cuda_number}")
        else:
            print(f"[Warning] cuda:{cuda_number} not available. Falling back to CPU.")
    return torch.device("cpu")

device = get_device(cuda_number=0)

# import jax
# jax.config.update("jax_disable_jit", True)

# @dataclasses.dataclass
# class Checkpoint:
#     """Load a policy from a trained checkpoint."""
#     # Training config name (e.g., "pi0_aloha_sim").
#     config: str
#     # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
#     dir: str

# @dataclasses.dataclass
# class Default:
#     """Use the default policy for the given environment."""

# @dataclasses.dataclass
# class Args:
#     """Arguments for the directly deploy pi0 script."""
#     # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
#     # prompt.
#     default_prompt: str | None = None
#     # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
#     policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# def create_policy(args: Args) -> _policy.Policy:
#     """Create a policy from the given arguments."""
    
#     print("args.policy:", args.policy)
#     return _policy_config.create_trained_policy(
#         _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
#     )


if __name__ == "__main__":
    
    # if not GlobalHydra.instance().is_initialized():
    #     hydra.initialize(config_path="configs")
    # cfg_wm = hydra.compose(config_name="wm_config", overrides=[])

    # world = WorldModel(device, cfg_wm)
    # ckpt_dir = "/home/luka/Wenkai/openvla/third_party/world_models/pick_bowl"
    # world.load_snapshot(ckpt_dir)
    # print("World model loaded successfully.")
    # task_config_path = ""
    # mcts_po = MCTS_RL(
    #     device=torch.device(device), 
    #     task_config_path=task_config_path
    # )
    # mcts_every_step = 5
    
    # single_action = (0.22, 0.123, 0.3, -0.035, 1, -0.0024, -0.00445, 1) # (0.333, 0.069, 0.24, 0.0049, -0.017, 0.9968, 0.0768, 1)
    pick_init_action = (0.283, 0.049, 0.15, 0.00, 0.00, 1, 0, 1)
    stack_init_action = (0.243, 0.069, 0.17, 0.00, 0.00, 1, 0, 1)
    press_init_action = (0.253, 0.15, 0.18, 0.00, 0.00, 1, 0, 1)
    
    
    # runhao:
    pick_bowl_action = (0.319, 0.110, 0.186, 0.00, 0.00, 1, 0, 1) 
    stack_cup_aciton = (0.388, 0.280, 0.27, 0.00, 0.00, 1, 0, 1) 
    pick_plate_aciton = (0.319, 0.096, 0.227, 0.00, 0.00, 1, 0, 1) 
    execute_action = pick_bowl_action
    
    pick_bowl_mean = (0.320,  0.061,  0.2456, 0.00, 0.00, 1, 0, 1)
    stack_cup_mean = (0.4, 0.26,  0.28, 0.00, 0.00, 1, 0, 1) # (0.3862, 0.296, 0.3333, -0.0958, 0.9948, -0.0185, -0.0304, 1) # (0.4, 0.26,  0.28, 0.00, 0.00, 1, 0, 1)
    pick_plate_mean = (0.394,  0.182,  0.257, 0.00, 0.00, 1, 0, 1)
    execute_action = stack_cup_mean
    execute_eef_IK(execute_action)
    # controller = A1ArmController(
    #     urdf_path="/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
    #     base_link="base_link",
    #     ee_link="arm_joint6"
    # )
    # controller.execute_ik_action(execute_action)
    
    from scipy.spatial.transform import Rotation as R
    def quaternion_to_rpy(quaternion):
        """
        将四元数转换为RPY欧拉角（roll, pitch, yaw）。

        param quaternion: 包含四个元素的列表或元组，表示四元数 (x, y, z, w)。
        return: 包含三个元素的列表，表示RPY欧拉角 (roll, pitch, yaw)，单位为弧度。
        """
        r = R.from_quat(quaternion)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return [roll, pitch, yaw]
    
 
    init_pose = execute_action[0:3]  # 初始位置
    init_quat = execute_action[3:7]  # 初始四元数
    init_rpy = quaternion_to_rpy(init_quat) 
    init_gripper = -execute_action[-1] 
    
    print("!!!Initial rpy:", init_rpy)

    pipelines = initialize_camera()
    print("Camera pipeline initialized.")
    for i in range(50):
        image = get_L515_image(pipelines) 
        if image is not None:
            image.save("captured_image.png") # BGR here, trained with 
        image = get_D435_image(pipelines) 
        if image is not None:
            image.save("captured_image_wrist.png") # BGR here, trained with 

    step=0
    
    
    # model_path = "/home/luka/Wenkai/openpi/pi0/runhao/29999" # ./checkpoints/pi0_your_custom_task/your_experiment_name/latest/params
     
    default_prompt = "stack the green cup and pink cup on the purple cup"  
    # # "Press the button of sanitizer."  "Stack green cube on blue cube."  "Pick green cube and put into the bowl."
    # # for pi0 runhao: 
    # #     place the green cube and orange into the bowl
    # #     place the banana and mango into the plate
    # #     stack the green cup and pink cup on the purple cup

    # args = Args(
    #     default_prompt=default_prompt,
    #     policy=Checkpoint(
    #         config="pi0_libero_low_mem_finetune_real_demo", # follow config.py
    #         dir=model_path
    #     )
    # )
    # policy = create_policy(args)

    # print("Policy loaded.")
    
    
    client = _websocket_client_policy.WebsocketClientPolicy("0.0.0.0", 8000)
    
    loop_interval = 0.1  # 控制频率
    
    
    

    try:
        while step < 200: # 60 for pick, 80 for stack, 50 for press
        # while True:
            print("Starting new iteration of the control loop...")
            start_time = time.time()
            
            try:
                # print("Attempting to get image from camera...")
                # image = get_frame(pipeline, target_width=384, target_height=384)
                image = get_L515_image(pipelines) 
                image_wrist = get_D435_image(pipelines)
                if image is not None:
                    # Convert the image to BGR format before saving
                    bgr_image = image.convert("RGB")  # Ensure it's in RGB first
                    bgr_image = bgr_image.split()[::-1]  # Reverse channels to BGR
                    bgr_image = Image.merge("RGB", bgr_image)
                    bgr_image.save(f"/home/luka/Wenkai/visualization/captured_image_{step}.png")
                    
                    bgr_image_wrist = image_wrist.convert("RGB")  # Ensure it's in RGB first
                    bgr_image_wrist = bgr_image_wrist.split()[::-1]  # Reverse channels to BGR
                    bgr_image_wrist = Image.merge("RGB", bgr_image_wrist)
                    bgr_image_wrist.save(f"/home/luka/Wenkai/visualization/captured_image_wrist_{step}.png")
                if image is None:
                    print("No image captured from camera.")
                    continue
                # print("Image acquired successfully.")
            except Exception as e:
                print(f"Failed to get image from camera: {e}")
                time.sleep(loop_interval)
                continue
            
            img_array = np.asarray(bgr_image).astype(np.uint8)[None, ...]
            img_wrist_array = np.asarray(bgr_image_wrist).astype(np.uint8)[None, ...]
            print(f"Image shape: {img_array.shape}")
            
            current_state = np.concatenate((init_pose, init_rpy, [init_gripper]), axis=0)
            element = {
                "observation/image": img_array[0],
                "observation/wrist_image": img_wrist_array[0],
                "observation/state": current_state,
                # "observation/state": np.zeros((7), dtype=np.float32),
                "prompt": default_prompt,
            }
            
            inf_time = time.time()
            # local model
            # action = policy.infer(element)
            
            # server 
            action = client.infer(element)
            print(f"-----------------------Inference time:{time.time() - inf_time} --------------------------")

            # print(f"!!!!!!!!!!!!!!!!!Predicted action: {action}")

            all_actions = action["actions"]   # (50, 7)

            # delta_actions = []
            # # 第一个 action 减去当前状态
            # delta_actions.append(all_actions[0] - current_state)

            # # 后续 action 两两相减
            # for i in range(1, all_actions.shape[0]):
            #     delta = all_actions[i] - all_actions[i - 1]
            #     delta_actions.append(delta)

            # # 转换为 numpy 数组
            # delta_actions = np.stack(delta_actions)
            # all_actions = delta_actions
            
            # print(f"all_actions: {all_actions}")

            # 执行前 n 步
            n_steps = 4
            n_steps = min(len(all_actions), n_steps) 
            merge_step = 2 
            
            for step_idx in range(0, n_steps - 1, merge_step):
                
                try:
                    # print("Attempting to get image from camera...")
                    # image = get_frame(pipeline, target_width=384, target_height=384)
                    image = get_L515_image(pipelines) 
                    if image is not None:
                        # Convert the image to BGR format before saving
                        bgr_image = image.convert("RGB")  # Ensure it's in RGB first
                        bgr_image = bgr_image.split()[::-1]  # Reverse channels to BGR
                        bgr_image = Image.merge("RGB", bgr_image)
                        bgr_image.save(f"/home/luka/Wenkai/visualization/captured_image_{step+step_idx}.png")
                    if image is None:
                        print("No image captured from camera.")
                        continue
                    # print("Image acquired successfully.")
                except Exception as e:
                    print(f"Failed to get image from camera: {e}")
                    time.sleep(loop_interval)
                    continue
                    
                merged_chunk = all_actions[step_idx : step_idx + merge_step]
                merged_action_prefix = np.sum(merged_chunk[:, 0:6], axis=0)
                gripper_command = merged_chunk[-1][6]
                action_step = np.concatenate([merged_action_prefix, [gripper_command]])
                
                current_position = init_pose
                current_rpy = init_rpy

                # 计算 delta_position、delta_rpy、gripper_command
                delta_position = [
                    v * 1 if abs(v) < 1e-4
                    else v * 1 if abs(v) < 1e-3
                    else v * 1 if abs(v) < 1e-2
                    else v
                    for v in action_step[0:3]
                ]
                # if abs(delta_position[2]) < 1e-3:
                #     delta_position[2] *= 10

                delta_rpy = action_step[3:6]
                gripper_command = action_step[6]

                # update target position and orientation
                new_position = [current_position[i] + delta_position[i] for i in range(3)]
                new_rpy = [current_rpy[i] + delta_rpy[i] for i in range(3)]

                # limit the range of new_position
                x_threshold = 0.45
                y_threshold_min, y_threshold_max = -0.03, 0.34
                z_threshold_min, z_threshold_max = 0.11, 0.35  # 0.11 for cup, 0.09 for others

                new_position[0] = min(new_position[0], x_threshold)
                new_position[1] = min(max(new_position[1], y_threshold_min), y_threshold_max)
                new_position[2] = min(max(new_position[2], z_threshold_min), z_threshold_max)

                init_pose = new_position
                
                
                init_rpy = [0.027, -0.01, 3.15] # new_rpy
                init_gripper = gripper_command


                gripper_position = (1 - gripper_command) / 2
                # if step_idx > 20 and step_idx % 5 == 0 and gripper_position > 0.8:
                #     gripper_position = 0

                final_action = new_position + new_rpy + [gripper_position]
                print(f"Step {step_idx} Final action:", final_action)
                assert len(final_action) == 7
                exe_time = time.time()

                execute_eef_IK(final_action)
                # controller.execute_ik_action(final_action)

                # time.sleep(1.0 / control_frequency)
            print(f"-----------------------Execution time:{time.time() - exe_time} --------------------------")
            # 移动机械臂到新的目标位姿
            # ur_robot.move_j_p(new_position + new_rpy, k_acc=0.5, k_vel=1.5)

            # 控制夹爪开合程度
            
            # try:
            #     # ur_robot.set_gripper_position(int(gripper_position))
            # except Exception as e:
            #     print(f"Failed to control gripper: {e}")
            #     continue

            # 维持控制频率
            elapsed_time = time.time() - start_time
            print(f"-----------------------total time:{elapsed_time} --------------------------")
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("Warning: Control loop is running slower than desired frequency")
            step += 1
            
        final_action = action[0:6] + [1]
        # execute_eef_IK(final_action)
        controller.execute_ik_action(final_action)
        time.sleep(1)
        # execute_eef_IK(execute_action)
        controller.execute_ik_action(execute_action)
        
                
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()