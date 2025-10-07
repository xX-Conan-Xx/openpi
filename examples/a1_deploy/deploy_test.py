"""
Minimal Test Script - 最小化模型测试
Author: Wenkai
"""

import time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import torch
from PIL import Image
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
from openpi_client import websocket_client_policy

class MinimalModelTester:
    
    def __init__(self, model_type="pi0", websocket_host="0.0.0.0", websocket_port=8000):
        self.model_type = model_type.lower()
        self.pipelines = None
        
        # 初始化模型客户端
        if self.model_type in ["pi0", "memo-vla"]:
            self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
            print(f"Initialized {self.model_type.upper()} client at {websocket_host}:{websocket_port}")
        
        # 设置默认状态和提示词
        self.default_state = np.array([  0.3959,  -0.0129,   0.2078,   3.1416,   0.0000,   3.1416,  -1.0000])  # down
        # ([ 0.2922,   0.2918,   0.3015,   3.1416,   0.0000,   3.1416,  -1.0000])  # down
        # 0.2832,   0.2916,   0.2463,   3.1416,   0.0000,   3.1416,  -1.0000 # up
        # 0.2866,   0.3018,   0.2730,   3.0021,   0.0000,   3.0013,  -1.0000 # left
        # 0.2932,   0.3354,   0.2977,   3.1416,   0.0000,   3.1416,  -1.0000 # right
        
        self.default_prompt = "Place the bread on the plate."
        # "Stack the cups together."
        # "Place the vegetables on the plate."
        # "Place the bread on the plate."
        # "Put the vegetables into the bowl."
        # "Place the fruit on the plate."
        # "Put the fruit into the bowl."
        # "Pick up the cup."
        
        
    def initialize_camera(self):
        """初始化相机"""
        self.pipelines = initialize_camera()
        print("Camera initialized")
        
        # 相机预热
        for _ in range(10):
            get_L515_image(self.pipelines)
            get_D435_image(self.pipelines)
        print("Camera warmed up")
    
    def capture_single_frame(self):
        """捕获单帧图像"""
        try:
            main_image = get_L515_image(self.pipelines)
            wrist_image = get_D435_image(self.pipelines)
            
            if main_image is None or wrist_image is None:
                print("Failed to capture images")
                return None, None
                
            # 转换为RGB格式
            main_image = main_image.convert("RGB")
            wrist_image = wrist_image.convert("RGB")
            
            print(f"Captured images - Main: {main_image.size}, Wrist: {wrist_image.size}")
            return main_image, wrist_image
            
        except Exception as e:
            print(f"Error capturing images: {e}")
            return None, None
    
    def prepare_model_input(self, main_image, wrist_image, state=None, prompt=None):
        """准备模型输入数据"""
        if state is None:
            state = self.default_state
        if prompt is None:
            prompt = self.default_prompt
            
        # 转换图像为numpy数组
        main_array = np.asarray(main_image).astype(np.uint8)
        wrist_array = np.asarray(wrist_image).astype(np.uint8)
        
        element = {
            "observation/image": main_array,
            "observation/wrist_image": wrist_array,
            "observation/state": state,
            "prompt": prompt,
        }
        
        print(f"Input prepared:")
        print(f"  Main image shape: {main_array.shape}")
        print(f"  Wrist image shape: {wrist_array.shape}")
        print(f"  State: {state}")
        print(f"  Prompt: {prompt}")
        
        return element
    
    def test_model_inference(self, element):
        """测试模型推理"""
        print("\n--- Starting Model Inference ---")
        
        start_time = time.time()
        
        try:
            if self.model_type in ["pi0", "memo-vla"]:
                result = self.client.infer(element)
            else:
                print(f"Model type {self.model_type} not supported in minimal test")
                return None
                
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.4f}s")
        
        return result
    
    def analyze_output(self, result):
        """分析模型输出"""
        print("\n--- Model Output Analysis ---")
        
        if result is None:
            print("No result to analyze")
            return
        
        if isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
            print(f"\nActions analysis:")
            print(f"  Type: {type(actions)}")
            print(f"  Length: {len(actions)}")
            
            if len(actions) > 0:
                actions_array = np.asarray(actions)
                print(f"  Shape: {actions_array.shape}")
                print(f"  Data type: {actions_array.dtype}")
            
                print(f"\nactions: {actions+self.default_state} ")
                
        
        else:
            print("No 'actions' key found in result")
            print(f"Full result: {result}")
    
    def run_single_test(self, use_camera=True, save_images=False):
        """运行单次测试"""
        print("=== Minimal Model Test ===")
        
        # 初始化相机或使用固定图像
        if use_camera:
            if self.pipelines is None:
                self.initialize_camera()
            
            # 捕获图像
            main_image, wrist_image = self.capture_single_frame()
            if main_image is None or wrist_image is None:
                print("Failed to capture images, aborting test")
                return
        else:
            # 使用固定图像进行测试
            try:
                main_image = Image.open('/home/luka/Wenkai/model_test_img/bread_obs_rgb_frame_000.png')
                wrist_image = Image.open('/home/luka/Wenkai/model_test_img/bread_wrist_frame_000.png')
                print("Using fixed test images")
            except Exception as e:
                print(f"Failed to load test images: {e}")
                return
        
        # 保存图像（可选）
        if save_images:
            main_image.save('/tmp/test_main_image.jpg')
            wrist_image.save('/tmp/test_wrist_image.jpg')
            print("Test images saved to /tmp/")
        
        # 准备模型输入
        element = self.prepare_model_input(main_image, wrist_image)
        
        # 测试模型推理
        result = self.test_model_inference(element)
        
        # 分析输出
        self.analyze_output(result)
        
        print("\n=== Test Completed ===")


def main():
    """主函数"""
    # 初始化测试器
    tester = MinimalModelTester(model_type="pi0")  # 可以改为 "memo-vla"
    
    # 运行测试
    tester.run_single_test(
        use_camera=False,      # 设为False使用固定图像
        save_images=False      # 保存测试图像
    )


if __name__ == "__main__":
    main()