"""
A1 Arm Deployment Template
Author: Wenkai (Template)
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


class A1ArmDeploy:
    """Template for A1 Arm deployment with multiple model support"""
    
    def __init__(self, controller, model_type="pi0", 
                 websocket_host="0.0.0.0", websocket_port=8000,
                 cogact_url='http://127.0.0.1:5500/api/inference',
                 openvla_path=None):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.controller = controller
        self.model_type = model_type.lower()
        self.pipelines = None
        
        # Default position limits (can be customized per task)
        self.pos_limits = {
            "x_min": 0.2, "x_max": 0.4,
            "y_min": -0.02, "y_max": 0.32,
            "z_min": 0.11, "z_max": 0.27
        }
        
        # Default task configuration
        self.default_prompt = "Pick and place task"
        self.default_preset = (0.3, 0.1, 0.2, 0.00, 0.00, 1.0, 0.0, 1.0)
        
        self._init_model_client(websocket_host, websocket_port, cogact_url, openvla_path)
        
    def _init_model_client(self, websocket_host, websocket_port, cogact_url, openvla_path):
        """Initialize model client based on model type"""
        if self.model_type == "pi0":
            self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
            print(f"Initialized PI0 client at {websocket_host}:{websocket_port}")
            
        elif self.model_type == "cogact":
            self.cogact_url = cogact_url
            print(f"Initialized CogACT client at {cogact_url}")
            
        elif self.model_type == "openvla":
            if openvla_path is None:
                raise ValueError("OpenVLA path must be provided for openvla model type")
            self.processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
            self.vla = AutoModelForVision2Seq.from_pretrained(
                openvla_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            print(f"Initialized OpenVLA from {openvla_path}")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def initialize_camera(self, warm_up_frames=10):
        """Initialize camera and warm up"""
        self.pipelines = initialize_camera()
        print("Camera pipeline initialized.")
        
        # Camera warm-up
        for _ in range(warm_up_frames):
            get_L515_image(self.pipelines)
            get_D435_image(self.pipelines)
        
    def capture_images(self, step=None, save_dir=None):
        """Capture images from cameras"""
        try:
            main_image = get_L515_image(self.pipelines)
            wrist_image = get_D435_image(self.pipelines)
            
            if main_image is None or wrist_image is None:
                print("Failed to capture images")
                return None, None, None
                
            bgr_main = self._convert_to_bgr(main_image)
            bgr_wrist = self._convert_to_bgr(wrist_image)
            
            main_path = None
            if step is not None and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"image_{step}.png")
                wrist_path = os.path.join(save_dir, f"wrist_{step}.png")
                bgr_main.save(main_path)
                bgr_wrist.save(wrist_path)
                
            return bgr_main, bgr_wrist, main_path
            
        except Exception as e:
            print(f"Failed to capture images: {e}")
            return None, None, None
    
    def _convert_to_bgr(self, image):
        """Convert image to BGR format"""
        rgb_image = image.convert("RGB")
        channels = rgb_image.split()[::-1]  
        return Image.merge("RGB", channels)
        
    def quaternion_to_rpy(self, quaternion):
        """Convert quaternion to roll-pitch-yaw"""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)
        
    def set_initial_position(self, position=None):
        """Set robot to initial position"""
        if position is None:
            position = self.default_preset
        self.controller.execute_eef(position, "init")
        return position
            
    def prepare_inference_data(self, main_image, wrist_image, current_state, prompt=None):
        """Prepare data for model inference"""
        return {
            "observation/image": np.asarray(main_image, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
            "observation/state": current_state,
            "prompt": prompt or self.default_prompt,
        }
        
    def apply_position_limits(self, position):
        """Apply position limits for safety"""
        position[0] = np.clip(position[0], self.pos_limits["x_min"], self.pos_limits["x_max"])
        position[1] = np.clip(position[1], self.pos_limits["y_min"], self.pos_limits["y_max"])
        position[2] = np.clip(position[2], self.pos_limits["z_min"], self.pos_limits["z_max"])
        return position
    
    def process_action(self, action_step, current_position, current_rpy):
        """Process action step and apply constraints"""
        # Apply position delta
        delta_position = action_step[:3]
        new_position = [current_position[i] + delta_position[i] for i in range(3)]
        new_position = self.apply_position_limits(new_position)
        
        # Apply orientation delta or use fixed orientation
        new_rpy = [0.027, -0.01, 3.15]  # Fixed orientation - can be customized
        
        # Process gripper command
        gripper_command = action_step[6]
        if self.model_type in ["pi0", "openvla"]:
            gripper_position = (1 - gripper_command) / 2
        else:  # cogact
            gripper_position = 1 - gripper_command
        
        return new_position, new_rpy, gripper_position
    
    def infer_action(self, main_image, wrist_image, current_state, prompt=None):
        """Perform model inference"""
        if self.model_type == "pi0":
            element = self.prepare_inference_data(main_image, wrist_image, current_state, prompt)
            return self.client.infer(element)
            
        elif self.model_type == "cogact":
            # Save image temporarily for CogACT
            temp_path = "/tmp/temp_image.png"
            main_image.save(temp_path)
            
            data = {'task_description': prompt or self.default_prompt}
            files = {
                'json': (None, json.dumps(data), 'application/json'),
                "images": (os.path.basename(temp_path), open(temp_path, "rb"), "image/png"),
            }
            response = requests.post(self.cogact_url, files=files)
            files["images"][1].close()  # Close file handle
            return response.json()
            
        elif self.model_type == "openvla":
            prompt_text = f"In: What action should the robot take to {(prompt or self.default_prompt).lower()}?\\nOut:"
            inputs = self.processor(prompt_text, main_image).to(self.device, dtype=torch.bfloat16)
            return self.vla.predict_action(**inputs, unnorm_key='default', do_sample=False)
    
    def execute_action_sequence(self, all_actions, chunk_size, merge_step, current_state):
        """Execute a sequence of actions"""
        n_steps = min(len(all_actions), chunk_size)
        init_pose, init_rpy, init_gripper = current_state[:3], current_state[3:6], current_state[6]
        
        for step_idx in range(0, n_steps, merge_step):
            # Merge actions if needed
            merged_chunk = all_actions[step_idx:step_idx + merge_step]
            
            if len(merged_chunk) > 1:
                # Merge position/orientation deltas, keep last gripper command
                merged_action_prefix = np.sum(merged_chunk[:, :6], axis=0)
                gripper_command = merged_chunk[-1][6]
                action_step = np.concatenate([merged_action_prefix, [gripper_command]])
            else:
                action_step = merged_chunk[0]
            
            # Process action
            new_pos, new_rpy, gripper = self.process_action(action_step, init_pose, init_rpy)
            final_action = new_pos + new_rpy + [gripper]
            
            print(f"Executing: {final_action}")
            self.controller.execute_eef(final_action, "execute")
            
            # Update state
            init_pose, init_rpy, init_gripper = new_pos, new_rpy, gripper
        
        return init_pose, init_rpy, init_gripper
    
    def run_control_loop(self, n_iterations=200, chunk_size=4, merge_step=2, 
                        loop_interval=0.1, save_images=False, save_dir=None):
        """Main control loop"""
        # Initialize camera if needed
        if self.pipelines is None:
            self.initialize_camera()
            
        # Set initial position
        initial_action = self.set_initial_position()
        init_pose = list(initial_action[:3])
        init_quat = initial_action[3:7]
        init_rpy = self.quaternion_to_rpy(init_quat)
        init_gripper = -initial_action[-1]
        
        print(f"Initial state - Pose: {init_pose}, RPY: {init_rpy}, Gripper: {init_gripper}")
        
        step = 0
        
        try:
            while step < n_iterations:
                print(f"\\n--- Step {step} ---")
                start_time = time.time()
                
                # Capture images
                main_image, wrist_image, main_path = self.capture_images(
                    step if save_images else None, save_dir
                )
                
                if main_image is None or wrist_image is None:
                    print("Skipping step due to image capture failure")
                    time.sleep(loop_interval)
                    continue
                
                # Prepare current state
                current_state = np.concatenate([init_pose, init_rpy, [init_gripper]])
                
                # Perform inference
                inf_start = time.time()
                action_result = self.infer_action(main_image, wrist_image, current_state)
                print(f"Inference time: {time.time() - inf_start:.4f}s")
                
                # Process actions based on model type
                if self.model_type in ["pi0"]:
                    all_actions = np.asarray(action_result["actions"])
                    init_pose, init_rpy, init_gripper = self.execute_action_sequence(
                        all_actions, chunk_size, merge_step, current_state
                    )
                else:
                    # Single-step models (CogACT, OpenVLA)
                    if isinstance(action_result, list):
                        action_step = action_result
                    else:
                        action_step = action_result.get("action", action_result)
                    
                    new_pos, new_rpy, gripper = self.process_action(action_step, init_pose, init_rpy)
                    final_action = new_pos + new_rpy + [gripper]
                    
                    print(f"Executing: {final_action}")
                    self.controller.execute_eef(final_action, "execute")
                    
                    init_pose, init_rpy, init_gripper = new_pos, new_rpy, gripper
                
                # Control loop timing
                elapsed_time = time.time() - start_time
                print(f"Step time: {elapsed_time:.4f}s")
                
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Loop running slower than desired frequency")
                    
                step += 1
                
        except KeyboardInterrupt:
            print("\\nControl loop interrupted by user")
        except Exception as e:
            import traceback
            print(f"Error occurred: {e}")
            traceback.print_exc()
        finally:
            print("Control loop ended")
            # Return to initial position
            self.controller.execute_eef(initial_action, "reset")


def main():
    """Example usage"""
    from controller_eef import A1ArmController
    
    # Initialize controller and robot system
    controller = A1ArmController()
    robot_system = A1ArmDeploy(
        controller, 
        model_type="pi0",  # Options: "pi0", "cogact", "openvla"
        websocket_host="0.0.0.0",
        websocket_port=8000,
        # openvla_path="/path/to/openvla/model"  # Required for OpenVLA
    )
    
    # Configure task-specific settings
    robot_system.default_prompt = "Your task description here"
    robot_system.pos_limits = {
        "x_min": 0.2, "x_max": 0.4,
        "y_min": -0.02, "y_max": 0.32,
        "z_min": 0.11, "z_max": 0.27
    }
    
    # Run control loop
    robot_system.run_control_loop(
        n_iterations=1000,
        chunk_size=5,
        merge_step=1,
        loop_interval=0.1,
        save_images=True,
        save_dir="/path/to/save/images"
    )


if __name__ == "__main__":
    main()