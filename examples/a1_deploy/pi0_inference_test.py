
import time
import numpy as np
import torch
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from openpi_client import websocket_client_policy


class PI0InferenceSystem:
    """PI0纯推理测试系统"""
    
    # 预设位置配置（用于状态输入）
    POSITION_PRESETS = {
        "place_fruit_bowl": (0.44, 0.0, 0.15, 0.00, 0, 1, 0, 1),
        "place_bread_plate": (0.422427, 0.002967, 0.190676, 0.998548, 0.027531, 0.030488, -0.034847, 1),
        "place_vege_plate": (0.40, 0.0, 0.24, 0.00, 0, 1, 0, 1),
        "put_vege_bowl": (0.42, 0.0, 0.13, 0.00, 0, 1, 0, 1),
        "pick_bowl": (0.320, 0.12, 0.2456, 0.00, 0.00, 1, 0, 1),
        "stack_cup": (0.4, 0.26, 0.28, 0.00, 0.00, 1, 0, 1),
        "pick_plate": (0.394, 0.182, 0.257, 0.00, 0.00, 1, 0, 1),
    }
    
    # 任务提示词
    TASK_PROMPTS = {
        "place_fruit_bowl": "Place the orange on the plate.",
        "place_bread_plate": "Place the coconut bread on the plate.",
        "place_vege_plate": "Place the cucumber on the plate.",
        "place_vege_bowl": "Put the carrot into the bowl.",
        "pick_bowl": "place the green cube and orange into the bowl",
        "stack_cup": "stack the green cup and pink cup on the purple cup",
        "pick_plate": "place the banana and mango into the plate",
    }
    
    def __init__(self, websocket_host="0.0.0.0", websocket_port=8000):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inference_history = []
        
        # 初始化PI0客户端
        self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
    
    def load_image(self, image_path, target_size=(224, 224)):
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if target_size:
                image = image.resize(target_size)
            return image
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None
    
    def quaternion_to_rpy(self, quaternion):
        """四元数转欧拉角"""
        r = R.from_quat(quaternion)
        return r.as_euler('xyz', degrees=False)
    
    def get_initial_state(self, task_name):
        if task_name not in self.POSITION_PRESETS:
            print(f"Warning: Task {task_name} not found, using default stack_cup")
            task_name = "stack_cup"
            
        preset = self.POSITION_PRESETS[task_name]
        init_pose = list(preset[0:3])
        init_quat = preset[3:7]
        init_rpy = self.quaternion_to_rpy(init_quat)
        init_gripper = -preset[-1]  # -1是开启
        
        return np.concatenate((init_pose, init_rpy, [init_gripper]), axis=0)
    
    def prepare_inference_data(self, main_image, wrist_image, current_state, prompt):

        img_array = np.asarray(main_image).astype(np.uint8)
        img_wrist_array = np.asarray(wrist_image).astype(np.uint8)
        
        return {
            "observation/image": img_array,
            "observation/wrist_image": img_wrist_array,
            "observation/state": current_state,
            "prompt": prompt,
        }
    
    def single_inference(self, main_image_path, wrist_image_path, 
                        current_state=None, task_name="stack_cup", 
                        prompt=None, save_result=True):
        print(f"\n=== Single Inference Test ===")
        print(f"Task: {task_name}")
        print(f"Main image: {main_image_path}")
        print(f"Wrist image: {wrist_image_path}")
        
        # 加载图像
        main_image = self.load_image(main_image_path)
        wrist_image = self.load_image(wrist_image_path)
        
        if main_image is None or wrist_image is None:
            print("Error: Failed to load images")
            return None
        
        # 设置状态和提示词
        if current_state is None:
            current_state = self.get_initial_state(task_name)
        
        if prompt is None:
            prompt = self.TASK_PROMPTS.get(task_name, self.TASK_PROMPTS["stack_cup"])
        
        print(f"Prompt: {prompt}")
        print(f"Current state: {current_state}")
        
        # 准备推理数据
        element = self.prepare_inference_data(main_image, wrist_image, current_state, prompt)
        
        # 执行推理
        start_time = time.time()
        try:
            action = self.client.infer(element)
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.4f}s")
            print(f"First few actions: {action['actions']}")
            
            # 保存结果
            if save_result:
                result = {
                    "task_name": task_name,
                    "prompt": prompt,
                    "current_state": current_state.tolist(),
                    "main_image_path": main_image_path,
                    "wrist_image_path": wrist_image_path,
                    "inference_time": inference_time,
                    "actions": action["actions"],
                    "timestamp": time.time()
                }
                self.inference_history.append(result)
            
            return {
                "success": True,
                "actions": action["actions"],
                "inference_time": inference_time,
                "action_length": len(action["actions"])
            }
            
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    


def main():
    """主函数示例"""
    # 初始化推理系统
    inference_system = PI0InferenceSystem(
        websocket_host="0.0.0.0",  # 模型服务器地址
        websocket_port=8000        # 模型服务器端口
    )
    
    # 示例1: 单次推理测试
    print("=== Example 1: Single Inference ===")
    result = inference_system.single_inference(
        main_image_path="/home/luka/Wenkai/scene_frame_0001.jpg",
        wrist_image_path="/home/luka/Wenkai/wrist_frame__0001.jpg",
        task_name="place_bread_plate"
    )
    
    if result and result["success"]:
        print(f"Inference successful! Got {result['action_length']} actions")
        

if __name__ == "__main__":
    main()