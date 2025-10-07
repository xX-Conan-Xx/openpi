"""
PI0 Robot Controller - Optimized Version
Author: Wenkai (Optimized)
"""

import time
import numpy as np
import torch
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
from openpi_client import websocket_client_policy

class PI0RobotController:
    """Simplified PI0 Robot Controller"""
    
    # Preset positions and task prompts
    POSITION_PRESETS = {
        "stack_cups": (0.236, 0.421, 0.285, 0.00, 0.00, 1.00, 0.00, 1.00),
    }
    
    TASK_PROMPTS = {
        "stack_cups": "Stack the green cup and pink cup on the blue cup.",
    }
    
    def __init__(self, controller, websocket_host="0.0.0.0", websocket_port=8000):
        self.controller = controller
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipelines = None
        
        # Set numpy print precision
        np.set_printoptions(precision=4, suppress=True)
        
        # Initialize PI0 client
        self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
        print(f"Initialized PI0 client with WebSocket at {websocket_host}:{websocket_port}")
    
    def initialize_camera(self, warm_up_frames=10):
        """Initialize cameras and warm up"""
        camera1, camera2 = 'f0210138', '332522071841'
        self.pipelines = initialize_camera(camera1, camera2)
        print("Camera pipeline initialized.")
        
        # Camera warm-up
        for _ in range(warm_up_frames):
            get_L515_image(self.pipelines)
            get_D435_image(self.pipelines)
    
    def capture_images(self, step=None, save_dir="/home/luka/Wenkai/visualization/"):
        """Capture and process images from cameras"""
        try:
            main_image = get_L515_image(self.pipelines)
            wrist_image = get_D435_image(self.pipelines)
            
            if main_image is None or wrist_image is None:
                print("Failed to capture image from camera.")
                return None, None, None
                
            # Convert to BGR and resize
            bgr_main = self._convert_to_bgr(main_image).resize((224, 224))
            bgr_wrist = self._convert_to_bgr(wrist_image).resize((224, 224))
            
            # Save images if step is provided
            main_path = None
            if step is not None:
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"captured_image_{step}.png")
                wrist_path = os.path.join(save_dir, f"captured_image_wrist_{step}.png")
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
        """Convert quaternion to roll-pitch-yaw angles"""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)
    
    def preset_position(self, task_name):
        """Set robot to preset position"""
        if task_name not in self.POSITION_PRESETS:
            print(f"Preset {task_name} not found")
            return False
        
        self.controller.execute_eef(self.POSITION_PRESETS[task_name], task_name)
        return True
    
    def prepare_inference_data(self, main_image, wrist_image, current_state, prompt):
        """Prepare data for inference"""
        return {
            "observation/image": np.asarray(main_image, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
            "observation/state": current_state,
            "prompt": prompt,
        }
    
    def process_action(self, action_step, current_position, current_rpy):
        """Process action step and calculate new position/orientation"""
        # Apply position deltas
        delta_position = action_step[:3]
        new_position = [round(current_position[i] + delta_position[i], 5) for i in range(3)]
        
        # Apply orientation deltas
        delta_rpy = action_step[3:6]
        new_rpy = [round(current_rpy[i] + delta_rpy[i], 5) for i in range(3)]
        
        # Process gripper command
        gripper_command = action_step[6]
        gripper_position = round((1 - gripper_command) / 2, 5)
        
        # Fixed orientation values
        new_rpy = [0.027, -0.01, 3.15]
        
        return new_position, new_rpy, gripper_position
    
    def execute_action_chunk(self, all_actions, chunk_size, merge_step, step, 
                           init_pose, init_rpy, init_gripper, task_name):
        """Execute a chunk of actions with merging"""
        n_steps = min(len(all_actions), chunk_size)
        current_pose, current_rpy, current_gripper = init_pose, init_rpy, init_gripper
        
        for step_idx in range(0, n_steps, merge_step):
            self.capture_images(step + step_idx)
            
            # Merge actions
            merged_chunk = all_actions[step_idx:step_idx + merge_step]
            merged_action_prefix = np.sum(merged_chunk[:, :6], axis=0)
            gripper_command = merged_chunk[-1][6]
            action_step = np.concatenate([merged_action_prefix, [gripper_command]])
            
            # Process and execute action
            new_pos, new_rpy, grip = self.process_action(action_step, current_pose, current_rpy)
            if grip < 0.3:
                grip = 0.0
            else:
                grip = 1.0
            print("new_gripper", grip)
            final_action = new_pos + new_rpy + [grip]
            
            print(f"Executing: {final_action}")
            self.controller.execute_eef(final_action, task_name)
            
        return new_pos, new_rpy, grip
    
    def run_control_loop(self, task_name="stack_cups", n_iterations=200, 
                        chunk_size=8, merge_step=2, loop_interval=0.1):
        """Main control loop"""
        # Initialize camera if needed
        if self.pipelines is None:
            self.initialize_camera()
        
        # Set preset position
        if not self.preset_position(task_name):
            return

        # Get task prompt
        prompt = self.TASK_PROMPTS.get(task_name, self.TASK_PROMPTS["stack_cups"])
        print(f"Task: {task_name}, Prompt: {prompt}")
        
        # Initialize robot state
        execute_action = self.POSITION_PRESETS[task_name]
        init_pose = list(execute_action[:3])
        init_quat_exe = execute_action[3:7]
        init_quat_infer = [execute_action[3], execute_action[5], execute_action[4], execute_action[6]]
        
        print(f"Initial pose: {init_pose}, Quaternion: {init_quat_exe}")
        
        init_rpy_exe = self.quaternion_to_rpy(init_quat_exe)
        init_rpy_infer = self.quaternion_to_rpy(init_quat_infer) 
        init_gripper = -execute_action[-1]
        
        print(f"Initial state - Pose: {init_pose}, RPY: {init_rpy_exe}, Gripper: {init_gripper}")
        
        step = 0
        
        try:
            while step < n_iterations:
                print(f"\n--- Step {step} ---")
                start_time = time.time()
                
                # Capture images
                main_image, wrist_image, _ = self.capture_images(step)
                
                if main_image is None or wrist_image is None:
                    print("Failed to capture camera images, skipping step")
                    time.sleep(loop_interval)
                    continue
                
                # Prepare current state
                current_state = np.concatenate((init_pose, init_rpy_infer, [init_gripper]))
                
                # Prepare inference data
                element = self.prepare_inference_data(main_image, wrist_image, current_state, prompt)
                
                print("current_state:", current_state)
                
                # Perform inference
                inf_time = time.time()
                action = self.client.infer(element)
                print(f"Inference time: {time.time() - inf_time:.4f}s")
                
                # Process actions
                all_actions = np.asarray(action["actions"])
                actions_to_execute = all_actions[:chunk_size]
                
                # Log absolute positions
                absolute_actions = current_state + actions_to_execute.tolist()
                print("Actions to execute (absolute positions):")
                for i, abs_action in enumerate(absolute_actions):
                    print(f"  Step {i+1}: {abs_action}")
                
                # Execute action chunk
                new_pose, new_rpy_infer, new_gripper = self.execute_action_chunk(
                    all_actions, chunk_size, merge_step, step,
                    init_pose, init_rpy_exe, init_gripper, task_name
                )
                # print("new_gripper", new_gripper)
                
                # Update state
                init_pose, init_rpy_exe, init_gripper = new_pose, new_rpy_infer, new_gripper
                
                # Control loop timing
                elapsed_time = time.time() - start_time
                print(f"Total step time: {elapsed_time:.4f}s")
                
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Control loop running slower than desired frequency")
                
                step += 1
                
        except KeyboardInterrupt:
            print("\nControl loop interrupted by user")
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            print("Control loop ended")
            # Return to initial position
            self.controller.execute_eef(execute_action, "reset")


def main():
    """Main function to initialize and run the robot controller"""
    from controller_eef import A1ArmController
    
    # Initialize controller and robot system
    controller = A1ArmController()
    robot_system = PI0RobotController(controller)
    
    # Run control loop
    robot_system.run_control_loop(
        task_name="stack_cups",
        n_iterations=1000,
        chunk_size=10,
        merge_step=1,
        loop_interval=0.1
    )


if __name__ == "__main__":
    main()