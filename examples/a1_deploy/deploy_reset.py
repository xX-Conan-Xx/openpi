"""
Demo Script
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
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
from openpi_client import websocket_client_policy
from transformers import AutoModelForVision2Seq, AutoProcessor
from memory import TrajMatcher, infer_action
import keyboard
import sys


class A1ArmDeploy:
    
    def __init__(self, controller, model_type="pi0", 
                 websocket_host="0.0.0.0", websocket_port=8000,
                 cogact_url='http://127.0.0.1:5500/api/inference',
                 openvla_path = "/home/luka/Wenkai/openvla-models/openvla-7b+pick_bowl_a1+b16+lr-0.0005+lora-r32+dropout-0.1--image_aug"
                 ):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.controller = controller
        self.model_type = model_type.lower()
        self.pipelines = None
        
        # task position initialization
        self.position_presets = {

            "pick_bowl": (0.320, 0.12, 0.2456, 0.00, 0.00, 1, 0, 1), # demo set
            "stack_cup": (0.08, 0.0, 0.24, 0.05, 0.05, 0.05, 0.05, 1),  # demo set
            # "stack_cup": (0.42, 0.11, 0.23, 0.00, 0.00, 1, 0, 1),
            "pick_plate": (0.394, 0.182, 0.257, 0.00, 0.00, 1, 0, 1),
            
        }
        
        self.pos_limits = {
            "x_min": 0.35, "x_max": 0.425,  #0.2/0.40 for demo pick bowl, 0.35/0.45 for demo stack cup
            # "x_min": 0.3, "x_max": 0.47,
            "y_min": -0.05, "y_max": 0.4, # -0.02/0.32 for demo "pick bowl", -0.05/0.35 for demo "stack cup"
            # "y_min": -0.15, "y_max": 0.4,
            "z_min": 0.13, "z_max": 0.35 # 0.09/0.27 for demo "pick bowl", 0.13/0.36 for demo "stack cup"
        }
        
        self.prompts = {
            "pick_bowl": "place the green cube and orange into the bowl",
            "stack_cup": "stack the green cup and pink cup on the purple cup",
            # "stack_cup": "Stack the cups together.",
            "pick_plate": "place the banana and mango into the plate",
        }
        self.default_prompt = "stack the green cup and pink cup on the purple cup"
        
        # # "Press the button of sanitizer."  "Stack green cube on blue cube."  "Pick green cube and put into the bowl."
        # # for pi0 runhao: 
        # #     place the green cube and orange into the bowl
        # #     place the banana and mango into the plate
        # #     stack the green cup and pink cup on the purple cup
        
        
        if self.model_type == "memo-vla":
            self.task_names = [
            'place the green cube and orange into the bowl', 
            'place the banana and mango into the plate', 
            'stack the green cup and pink cup on the purple cup'
            ]
            self.task_index = self.task_names.index(self.default_prompt)
            self.traj_dir = "/home/luka/Wenkai/openpi/examples/a1_deploy/trajectory_info_real_demo/"+self.default_prompt
            states_json_path = os.path.join(self.traj_dir, "states.json")
            actions_json_path = os.path.join(self.traj_dir, "actions.json")
            segments_json_path = os.path.join(self.traj_dir, "segments.json")

            with open(states_json_path, "r", encoding="utf-8") as f:
                self.states_ref = json.load(f)
            with open(actions_json_path, "r", encoding="utf-8") as f:
                self.actions_ref = json.load(f)
            with open(segments_json_path, "r", encoding="utf-8") as f:
                self.segments_ref = json.load(f)

            print(self.segments_ref[0])
            print(len(self.segments_ref))

                
        self._init_model_client(websocket_host, websocket_port, cogact_url, openvla_path)
        
    def _init_model_client(self, websocket_host, websocket_port, cogact_url, openvla_path):
        
        if self.model_type == "pi0" or self.model_type == "memo-vla":
            self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
            print(f"Initialized PI0 client with WebSocket at {websocket_host}:{websocket_port}")
            
        elif self.model_type == "cogact":
            self.cogact_url = cogact_url
            self.requests = requests
            print(f"Initialized CogACT client with API URL: {cogact_url}")
            
        elif self.model_type == "openvla":
            self.processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
            self.vla = AutoModelForVision2Seq.from_pretrained(
                openvla_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            print(f"Initialized local OpenVLA from {openvla_path}")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Choose 'pi0' or 'cogact' or 'openvla'.")
    
    def initialize_camera(self):
        camera1, camera2 = 'f0210138', '332522071841'
        self.pipelines = initialize_camera(camera1, camera2)
        print("Camera pipeline initialized.")
        self._warm_up_camera()
        
    def _warm_up_camera(self, frames=10):
        for _ in range(frames):
            get_L515_image(self.pipelines)
            get_D435_image(self.pipelines)
        
    def capture_images(self, step=None):
        try:
            main_image = get_L515_image(self.pipelines)
            wrist_image = get_D435_image(self.pipelines)
            
            if main_image is None or wrist_image is None:
                print("Failed to capture image from camera.")
                return None, None
                
            bgr_main = self._convert_to_bgr(main_image)
            bgr_wrist = self._convert_to_bgr(wrist_image)
            
            if step is not None:
                save_dir = "/home/luka/Wenkai/visualization/"
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"captured_image_{step}.png")
                wrist_path = os.path.join(save_dir, f"captured_image_wrist_{step}.png")
                bgr_main.save(main_path)
                bgr_wrist.save(wrist_path)
                
            return bgr_main, bgr_wrist, main_path
            
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
            
    def element_retrieve(self, main_image, wrist_image, current_state, prompt=None):
        # pi prepare
        img_array = np.asarray(main_image).astype(np.uint8)[None, ...]
        img_wrist_array = np.asarray(wrist_image).astype(np.uint8)[None, ...]
        
        return {
            "observation/image": img_array[0],
            "observation/wrist_image": img_wrist_array[0],
            "observation/state": current_state,
            "prompt": prompt or self.default_prompt,
        }
        
    def process_action(self, action_step, current_position, current_rpy, task_name):
        
        delta_position = action_step[0:3]
        new_position = [current_position[i] + delta_position[i] for i in range(3)]
        if task_name == "pick_bowl":
            self.pos_limits["x_min"] = 0.2
            self.pos_limits["x_max"] = 0.4
            self.pos_limits["y_min"] = -0.02
            self.pos_limits["y_max"] = 0.32
            self.pos_limits["z_min"] = 0.09
            self.pos_limits["z_max"] = 0.29
        new_position[0] = min(max(new_position[0], self.pos_limits["x_min"]),
                                self.pos_limits["x_max"])
        new_position[1] = min(max(new_position[1], self.pos_limits["y_min"]), 
                                self.pos_limits["y_max"])
        new_position[2] = min(max(new_position[2], self.pos_limits["z_min"]), 
                                self.pos_limits["z_max"])
        
        delta_rpy = action_step[3:6]
        new_rpy = [0.027, -0.01, 3.15]  # 固定姿态，可修改为 current_rpy + delta_rpy
        
        gripper_command = action_step[6]
        if self.model_type in ["pi0","memo-vla","openvla"]:
            gripper_position = (1 - gripper_command) / 2
        if self.model_type in ["cogact"] :
            gripper_position = 1 - gripper_command

        
        return new_position, new_rpy, gripper_position
    
    def run_control_loop(self, n_iterations=200, chunk_size=4, merge_step=2, loop_interval=0.1):
        if self.pipelines is None:
            self.initialize_camera()

        task_name = "stack_cup"                                                                             # "pick_bowl", "stack_cup", "pick_plate"
        # "swap_init", "pick_init", "stack_init", "press_init"
        # "pick_bowl", "stack_cup", "pick_plate"
        # "pick_bowl_mean", "stack_cup_mean", "pick_plate_mean"
        # to show the environment: "env_show"
        
        self.preset(task_name) #  preset the initial position from the task
        # prompt=self.prompts[task_name]
        prompt = self.prompts.get(task_name)
        print(f"Prompt: {prompt}")
        
        self.action_history = []
                
        execute_action = self.position_presets[task_name]
        init_pose = execute_action[0:3]
        init_quat = execute_action[3:7]
        init_rpy = self.quaternion_to_rpy(init_quat)
        init_gripper = -execute_action[-1]
        
        print("Initial RPY:", init_rpy)
        
        step = 0
        if self.model_type == "memo-vla":
            matcher = TrajMatcher(self.states_ref, self.segments_ref, self.actions_ref)
            # planner = TrajPlanner(matcher, self.client, self.default_prompt, self.task_index)
            current_segment = 0
            states = []
        
        try:
            while step < n_iterations:
                print(f"\n--- Step {step} ---")
                start_time = time.time()
                
                main_image, wrist_image, main_path = self.capture_images(step)
                if main_image is None or wrist_image is None:
                    time.sleep(loop_interval)
                    continue
                
                current_state = np.concatenate((init_pose, init_rpy, [init_gripper]), axis=0)
                
                if self.model_type == "pi0":
                    element = self.element_retrieve(main_image, wrist_image, current_state, prompt)
                    inf_time = time.time()
                    action = self.client.infer(element) # pi0 & CogACT
                    
                if self.model_type == "memo-vla":
                    inf_time = time.time()
                    action = infer_action(step,
                          main_image, wrist_image,
                          current_state, states,            # 传入历史
                          prompt,
                          self.client, matcher,
                          self.task_index, current_segment)

                if self.model_type == "cogact":
                    files = {
                    'json': (None, json.dumps(self.data), 'application/json'),
                    "images": (os.path.basename(main_path), open(main_path, "rb"), "image/png"), 
                    }
                    inf_time = time.time()
                    action = self.requests.post(self.cogact_url, files=files).json()

                if self.model_type == "openvla":
                    prompt = f"In: What action should the robot take to {self.default_prompt.lower()}?\nOut:"
                    inf_time = time.time()
                    inputs = self.processor(prompt, main_image).to(self.device, dtype=torch.bfloat16)
                    action = self.vla.predict_action(**inputs, unnorm_key='pick_bowl_a1', do_sample=False) 

                # print(f"Action: {action}")
                print(f"Inference time: {time.time() - inf_time:.4f}s")
                
                # pi0
                # all_actions = action["actions"]
                
                # CogACT
                
                if self.model_type in ["pi0", "memo-vla"]:
                    # action["actions"]: List[List[float]] (chunk)
                    all_actions = action["actions"]
                    # if not all_actions:
                    #     print("No actions returned from the model.")
                    #     continue
                    all_actions = np.asarray(all_actions[0:])  # chunk: shape (N, 7)

                else:  # cogact => single-step list[float]
                    if self.model_type in ["cogact"] and not isinstance(action, list):
                        print(f"Unexpected action format: {action}")
                        continue
                    all_actions = [action]  # -> shape (1, 7)
                    all_actions = np.asarray(all_actions, dtype=np.float32)
                    
                # if isinstance(all_actions[0], (int, float)):
                #     all_actions = [all_actions]          # -> list[list[float]
                # all_actions = np.asarray(all_actions, dtype=np.float32)  # shape (N, 7)
                
                n_steps = min(len(all_actions), chunk_size)
                # print(f"inferred action: {all_actions}")
                
                
                if self.model_type in ["pi0", "memo-vla"]:
                    n_steps = min(len(all_actions), chunk_size)
                    # print(f"chunk action: {all_actions}")

                    for step_idx in range(0, n_steps, merge_step):
                        self.capture_images(step + step_idx)
                        
                        merged_chunk = all_actions[step_idx : step_idx + merge_step]
                        merged_action_prefix = np.sum(merged_chunk[:, 0:6], axis=0)
                        gripper_command = merged_chunk[-1][6]
                        action_step = np.concatenate([merged_action_prefix, [gripper_command]])
                        
                        # debug: when gripper open to close, do not stack action

                        new_pos, new_rpy, grip = self.process_action(action_step, init_pose, init_rpy, task_name)

                        if self.model_type == "memo-vla":
                            inter_pos = (np.asarray(new_pos) + np.asarray(init_pose)) * 0.5
                            inter_rpy = (np.asarray(new_rpy) + np.asarray(init_rpy)) * 0.5
                            states.append(np.concatenate([inter_pos, inter_rpy, [init_gripper]]))

                            final_rpy = [0.027, -0.01, 3.15]
                            states.append(np.concatenate([new_pos, final_rpy, [gripper_command]]))

                        if task_name =="pick_bowl" :
                            if step > 6 and step <12 and new_pos[1]>0.18:
                                new_pos[1]=0.18
                            if step > 30 and step < 40 and new_pos[0]>0.33:
                                new_pos[0]=0.33
                            if step > 48 and grip < 0.1 and new_pos[2] < 0.20:
                                new_pos[2] = 0.23
                                
                            # TODO: debug the open (force to open )
                            if step >65 and grip < 0.1 and new_pos[0] >0.37 and new_pos[1] >0.305:
                                grip = 1
                                
                            # if step >65 and grip < 0.1 and new_pos[0] >0.37 and new_pos[1] >0.31:
                            #     grip = 1
                            
                        if task_name == "stack_cup":
                            if step == 8 and grip > 0.9 and new_pos[2] > 0.21:
                                new_pos[2] = 0.19

                            if step > 9 and step < 40 and grip < 0.1 and new_pos[2] > 0.33:
                                new_pos[2] = 0.32
                                
                            if step > 25 and step < 55 and grip > 0.9:
                                if new_pos[1] > 0.16:
                                    new_pos[1] = 0.15  
                                    
                            if step > 40 and new_pos[1] < 0.1 and grip > 0.9:
                                if new_pos[1] > 0:
                                    new_pos[1] = -0.03
                            if step > 40 and new_pos[1] < 0.1 and grip > 0.9:
                                if new_pos[2] > 0.28:
                                    new_pos[2] = 0.26
                                    
                            if step > 45 and grip > 0.9:
                                if new_pos[2] > 0.16 and new_pos[2] < 0.23:
                                    new_pos[2] = 0.15  
                            
                            if step > 55 and new_pos[1] > 0.185 and grip < 0.1:
                                new_pos[1] = 0.18
                                    
                            if step > 60 and new_pos[1] < 0.0 and grip < 0.1 and new_pos[2] > 0.34:
                                new_pos[1] = 0.03
                            
                                    
                            # if step > 40 and new_pos[1] < 0.1 and grip > 0.9:

                            
                        init_pose, init_rpy, init_gripper = new_pos, new_rpy, grip
                        final_action = new_pos + new_rpy + [grip]
                        
                        if keyboard.is_pressed('space'):
                            reset_pose = (0.08, 0.0, 0.2456, 0.05, 0.05, 0.05, 0.05, 1)
                            self.controller.execute_eef(reset_pose, "reset")
                            sys.exit(0)
                        
                        print(f"Executing: {final_action}")
                        self.controller.execute_eef(final_action, task_name)

                elapsed_time = (time.time() - start_time) * 1000  
                print(f"Total step time: {elapsed_time:.4f} ms")
                
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Control loop is running slower than desired frequency")
                    
                step += 1
                
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
    robot_system = A1ArmDeploy(controller, model_type="memo-vla") # cogact or pi0 or memo-vla or openvla
    
    robot_system.run_control_loop(
        n_iterations=1000,  # 总迭代次数
        chunk_size=5,  # 推理后保留的步数
        merge_step=1,      # 合并的步数, 1 for CogACT ONLY
        loop_interval=0.1  # 控制循环间隔
    )


if __name__ == "__main__":
    main()