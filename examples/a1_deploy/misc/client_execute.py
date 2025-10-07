import requests
import json

import os
from PIL import Image
import numpy as np

import sys
sys.path.append('/home/luka/Wenkai/openpi/examples/a1_deploy')
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from pyrealsense_image import initialize_camera, stop_camera, get_L515_image, get_D435_image  
from eef_control import execute_eef_IK
import time

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
    


if __name__ == "__main__":

    execute_action= (0.444, 0.083, 0.175,0,0,1,0,1)# (0.34, 0.161, 0.242, 0.00, 0.00, 1, 0, 1)
    execute_eef_IK(execute_action)
 
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
    
    
    # server initialization
    url = 'http://127.0.0.1:5500/api/inference'

    # Prepare task description
    data = {
        'task_description': "swap the slots of two objects"
    }
        
    loop_interval = 0.1  # 控制频率
        

    try:
        while step < 500: 
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
                    
                    image_path = f"/home/luka/Wenkai/visualization/captured_image_{step}.png"
                    image_wrist_path = f"/home/luka/Wenkai/visualization/captured_image_wrist_{step}.png"
                    
                if image is None:
                    print("No image captured from camera.")
                    continue
                print("Image acquired successfully.")
            except Exception as e:
                print(f"Failed to get image from camera: {e}")
                time.sleep(loop_interval)
                continue
            
            data = {
                'task_description': "swap the slots of two objects"
            }
            
            files = {
                'images': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png'),
                'json': (None, json.dumps(data), 'application/json'),
            }

            action = requests.post(url, files=files)
            print(f"answer: {action}")

            if action.status_code == 200:
                print("Success:")
                action_list = action.json()
                print(action_list)
            else:
                print("Failed to get a response from the API")
                print(action.status_code, action.text)
                
            inf_time = time.time()

            print(f"-----------------------Inference time:{time.time() - inf_time} --------------------------")

            # all_actions = action["actions"]   # (50, 7)
            # n_steps = 4
            # n_steps = min(len(all_actions), n_steps) 
            # merge_step = 2 
            
            # for step_idx in range(0, n_steps - 1, merge_step):
                
            #     try:
            #         # print("Attempting to get image from camera...")
            #         # image = get_frame(pipeline, target_width=384, target_height=384)
            #         image = get_L515_image(pipelines) 
            #         if image is not None:
            #             # Convert the image to BGR format before saving
            #             bgr_image = image.convert("RGB")  # Ensure it's in RGB first
            #             bgr_image = bgr_image.split()[::-1]  # Reverse channels to BGR
            #             bgr_image = Image.merge("RGB", bgr_image)
            #             bgr_image.save(f"/home/luka/Wenkai/visualization/captured_image_{step+step_idx}.png")
            #         if image is None:
            #             print("No image captured from camera.")
            #             continue
            #         # print("Image acquired successfully.")
            #     except Exception as e:
            #         print(f"Failed to get image from camera: {e}")
            #         time.sleep(loop_interval)
            #         continue
                    
            #     merged_chunk = all_actions[step_idx : step_idx + merge_step]
            #     merged_action_prefix = np.sum(merged_chunk[:, 0:6], axis=0)
            #     gripper_command = merged_chunk[-1][6]
            #     action_step = np.concatenate([merged_action_prefix, [gripper_command]])
                
            action_step = np.array(action_list, dtype=np.float32)
            current_position = init_pose
            current_rpy = init_rpy

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
            gripper_command = -1 if action_step[6] == 0 else action_step[6] # only for CogACT


            # update target position and orientation
            new_position = [current_position[i] + delta_position[i] for i in range(3)]
            new_rpy = [current_rpy[i] + delta_rpy[i] for i in range(3)]

            # limit the range of new_position
            x_threshold_min, x_threshold_max = 0.27, 0.45
            y_threshold_min, y_threshold_max = -0.03, 0.34
            z_threshold_min, z_threshold_max = 0.09, 0.35  # 0.11 for cup, 0.09 for others

            new_position[0] = min(max(new_position[0], x_threshold_min), x_threshold_max)
            new_position[1] = min(max(new_position[1], y_threshold_min), y_threshold_max)
            new_position[2] = min(max(new_position[2], z_threshold_min), z_threshold_max)


            init_pose = new_position
            
            
            init_rpy = [0.027, -0.01, 3.15] # new_rpy
            init_gripper = gripper_command


            gripper_position = (1 - gripper_command) / 2

            final_action = new_position + new_rpy + [gripper_position]
            step_idx = step # only for CogACT
            print(f"Step {step_idx} Final action:", final_action)
            assert len(final_action) == 7
            exe_time = time.time()

            execute_eef_IK(final_action)

            print(f"-----------------------Execution time:{time.time() - exe_time} --------------------------")
            elapsed_time = time.time() - start_time
            print(f"-----------------------total time:{elapsed_time} --------------------------")
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("Warning: Control loop is running slower than desired frequency")
            step += 1
            
        final_action = action[0:6] + [1]
        execute_eef_IK(final_action)
        time.sleep(1)
        execute_eef_IK(execute_action)
                
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    