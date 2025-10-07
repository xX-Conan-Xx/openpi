"""
PI0 VLA模型泛化性测试脚本 - Generalization Test
Author: Wenkai (Modified)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4, suppress=True)
import torch
from PIL import Image
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
from openpi_client import websocket_client_policy

class GeneralizationTester:
    
    def __init__(self, model_type="pi0", websocket_host="0.0.0.0", websocket_port=8000):
        self.model_type = model_type.lower()
        self.pipelines = None
        
        # 初始化模型客户端
        if self.model_type in ["pi0", "memo-vla"]:
            self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
            print(f"Initialized {self.model_type.upper()} client at {websocket_host}:{websocket_port}")
        
        # 设置默认状态
        self.default_state = np.array([0.3, -0.0129, 0.2078, 3.1416, 0.0000, 3.1416, -1.0000])
        
        # 定义所有测试指令
        self.test_prompts = [
            "Place the bread on the plate.",
            "Stack the cups together.", 
            "Place the vegetables on the plate.",
            "Put the vegetables into the bowl.",
            "Place the fruit on the plate.",
            "Put the fruit into the bowl.",
            "Pick up the cup."
        ]
        
        # 存储结果
        self.results = {}
        
        # 创建输出目录
        self.output_dir = "/home/luka/Wenkai/figs/generalization_test"
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            
        # 转换图像为numpy数组
        main_array = np.asarray(main_image).astype(np.uint8)
        wrist_array = np.asarray(wrist_image).astype(np.uint8)
        
        element = {
            "observation/image": main_array,
            "observation/wrist_image": wrist_array,
            "observation/state": state,
            "prompt": prompt,
        }
        
        return element
    
    def test_model_inference(self, element):
        """测试模型推理"""
        start_time = time.time()
        
        try:
            if self.model_type in ["pi0", "memo-vla"]:
                result = self.client.infer(element)
            else:
                print(f"Model type {self.model_type} not supported")
                return None
                
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.4f}s")
        
        return result
    
    def extract_xyz_actions(self, result):
        """提取XYZ坐标动作"""
        if result is None:
            return None
            
        if isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
            actions_array = np.array(actions)
            
            print(f"Actions shape: {actions_array.shape}")
            
            # 处理 [50, 7] 形状的动作序列，只取前3列（XYZ坐标）
            if actions_array.ndim == 2 and actions_array.shape[1] >= 3:
                # 取所有时间步的XYZ坐标 [50, 3]
                xyz_sequence = actions_array[:, :3]
                # 每一行都加上default_state的前3个值
                xyz_sequence_with_offset = xyz_sequence + self.default_state[:3]
                print(f"Extracted XYZ sequence shape: {xyz_sequence_with_offset.shape}")
                print(f"Raw first XYZ: {xyz_sequence[0]}")
                print(f"Offset first XYZ: {xyz_sequence_with_offset[0]}")
                print(f"Raw last XYZ: {xyz_sequence[-1]}")
                print(f"Offset last XYZ: {xyz_sequence_with_offset[-1]}")
                return xyz_sequence_with_offset
            elif actions_array.ndim == 1 and len(actions_array) >= 3:
                # 如果是一维数组，取前3个
                xyz_actions = actions_array[:3].reshape(1, -1) + self.default_state[:3]
                print(f"Extracted XYZ from 1D array: {xyz_actions}")
                return xyz_actions
            else:
                print(f"Unexpected actions format: shape {actions_array.shape}")
                return None
        
        return None
    
    def run_generalization_test(self, use_camera=False):
        """运行泛化性测试"""
        print("=== PI0 VLA Generalization Test ===")
        
        # 初始化相机或使用固定图像
        if use_camera:
            if self.pipelines is None:
                self.initialize_camera()
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
        
        print(f"\nTesting {len(self.test_prompts)} different prompts:")
        print("=" * 60)
        
        # 对每个指令进行测试
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n[Test {i}/{len(self.test_prompts)}] Prompt: '{prompt}'")
            print("-" * 40)
            
            # 准备模型输入
            element = self.prepare_model_input(main_image, wrist_image, prompt=prompt)
            
            # 测试模型推理
            result = self.test_model_inference(element)
            
            # 提取XYZ动作
            xyz_actions = self.extract_xyz_actions(result)
            
            if xyz_actions is not None:
                self.results[prompt] = xyz_actions
                print(f"XYZ sequence shape: {xyz_actions.shape}")
                print(f"First XYZ: [{xyz_actions[0, 0]:.4f}, {xyz_actions[0, 1]:.4f}, {xyz_actions[0, 2]:.4f}]")
                print(f"Last XYZ:  [{xyz_actions[-1, 0]:.4f}, {xyz_actions[-1, 1]:.4f}, {xyz_actions[-1, 2]:.4f}]")
            else:
                print("Failed to extract XYZ actions")
                self.results[prompt] = None
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
        # 生成3D可视化
        self.create_3d_visualization()
        
        # 保存结果摘要
        self.save_results_summary()
    
    def create_3d_visualization(self):
        """创建3D可视化图表"""
        print("\nCreating 3D visualization...")
        
        # 准备数据
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if len(valid_results) == 0:
            print("No valid results to visualize")
            return
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个指令绘制轨迹
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
        
        for i, (prompt, xyz_sequence) in enumerate(valid_results.items()):
            # xyz_sequence 是 [50, 3] 的形状
            x_coords = xyz_sequence[:, 0]
            y_coords = xyz_sequence[:, 1] 
            z_coords = xyz_sequence[:, 2]
            
            # 绘制轨迹线
            ax.plot(x_coords, y_coords, z_coords, 
                   color=colors[i], alpha=0.7, linewidth=2.5, 
                   label=prompt)
            
            # 标记起始点和结束点
            start_scatter = ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                                     color=colors[i], s=150, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            end_scatter = ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                                   color=colors[i], s=150, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            
            # 添加起始点和结束点的文字标注
            ax.text(x_coords[0], y_coords[0], z_coords[0], '  begin', 
                   fontsize=10, color='darkred', fontweight='bold')
            ax.text(x_coords[-1], y_coords[-1], z_coords[-1], '  end', 
                   fontsize=10, color='darkblue', fontweight='bold')
        
        # 设置轴标签
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_zlabel('Z Position (m)', fontsize=14)
        
        # 设置标题
        ax.set_title('PI0 VLA Model Generalization Test\nAction Trajectories for Different Prompts', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例 - 更大字体和更好的位置
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, 
                          frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 保存图片
        output_path = os.path.join(self.output_dir, 'var_pose_var_prompt_3d.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D trajectory visualization saved to: {output_path}")
        
        # 创建2D投影图
        self.create_2d_projections(valid_results)
        
        plt.show()
    
    def create_2d_projections(self, valid_results):
        """创建2D投影图"""
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
        
        for i, (prompt, xyz_sequence) in enumerate(valid_results.items()):
            x_coords = xyz_sequence[:, 0]
            y_coords = xyz_sequence[:, 1]
            z_coords = xyz_sequence[:, 2]
            
            # XY平面投影
            axes[0].plot(x_coords, y_coords, color=colors[i], alpha=0.7, linewidth=2.5, label=prompt)
            axes[0].scatter(x_coords[0], y_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[0].scatter(x_coords[-1], y_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            # 添加begin/end标注
            axes[0].annotate('begin', (x_coords[0], y_coords[0]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkred', fontweight='bold')
            axes[0].annotate('end', (x_coords[-1], y_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkblue', fontweight='bold')
            axes[0].set_xlabel('X Position (m)', fontsize=12)
            axes[0].set_ylabel('Y Position (m)', fontsize=12)
            axes[0].set_title('XY Plane Projection', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # XZ平面投影
            axes[1].plot(x_coords, z_coords, color=colors[i], alpha=0.7, linewidth=2.5, label=prompt)
            axes[1].scatter(x_coords[0], z_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[1].scatter(x_coords[-1], z_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            # 添加begin/end标注
            axes[1].annotate('begin', (x_coords[0], z_coords[0]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkred', fontweight='bold')
            axes[1].annotate('end', (x_coords[-1], z_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkblue', fontweight='bold')
            axes[1].set_xlabel('X Position (m)', fontsize=12)
            axes[1].set_ylabel('Z Position (m)', fontsize=12)
            axes[1].set_title('XZ Plane Projection', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # YZ平面投影
            axes[2].plot(y_coords, z_coords, color=colors[i], alpha=0.7, linewidth=2.5, label=prompt)
            axes[2].scatter(y_coords[0], z_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[2].scatter(y_coords[-1], z_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            # 添加begin/end标注
            axes[2].annotate('begin', (y_coords[0], z_coords[0]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkred', fontweight='bold')
            axes[2].annotate('end', (y_coords[-1], z_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, color='darkblue', fontweight='bold')
            axes[2].set_xlabel('Y Position (m)', fontsize=12)
            axes[2].set_ylabel('Z Position (m)', fontsize=12)
            axes[2].set_title('YZ Plane Projection', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        
        # 在第一个子图下方添加大的图例
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12, 
                  frameon=True, fancybox=True, shadow=True)
        
        # 保存2D投影图
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
        output_path_2d = os.path.join(self.output_dir, 'var_pose_var_prompt_3d.png')
        plt.savefig(output_path_2d, dpi=300, bbox_inches='tight')
        print(f"2D trajectory projections saved to: {output_path_2d}")
    
    def save_results_summary(self):
        """保存结果摘要"""
        summary_path = os.path.join(self.output_dir, 'generalization_test_results.txt')
        
        with open(summary_path, 'w') as f:
            f.write("PI0 VLA Model Generalization Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.model_type.upper()}\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Default State: {self.default_state}\n\n")
            
            f.write("Results for each prompt:\n")
            f.write("-" * 30 + "\n")
            
            for i, (prompt, xyz_sequence) in enumerate(self.results.items(), 1):
                f.write(f"{i}. Prompt: '{prompt}'\n")
                if xyz_sequence is not None:
                    f.write(f"   Trajectory shape: {xyz_sequence.shape}\n")
                    f.write(f"   Start XYZ: [{xyz_sequence[0, 0]:.6f}, {xyz_sequence[0, 1]:.6f}, {xyz_sequence[0, 2]:.6f}]\n")
                    f.write(f"   End XYZ:   [{xyz_sequence[-1, 0]:.6f}, {xyz_sequence[-1, 1]:.6f}, {xyz_sequence[-1, 2]:.6f}]\n")
                else:
                    f.write("   Trajectory: Failed to extract\n")
                f.write("\n")
            
            # 添加统计信息
            valid_results = {k: v for k, v in self.results.items() if v is not None}
            if valid_results:
                # 统计所有轨迹的起始点和结束点
                start_points = np.array([xyz[0] for xyz in valid_results.values()])
                end_points = np.array([xyz[-1] for xyz in valid_results.values()])
                
                f.write("Statistical Summary:\n")
                f.write("-" * 20 + "\n")
                f.write("Start Points:\n")
                f.write(f"  Mean: [{np.mean(start_points[:, 0]):.6f}, {np.mean(start_points[:, 1]):.6f}, {np.mean(start_points[:, 2]):.6f}]\n")
                f.write(f"  Std:  [{np.std(start_points[:, 0]):.6f}, {np.std(start_points[:, 1]):.6f}, {np.std(start_points[:, 2]):.6f}]\n")
                f.write("End Points:\n")
                f.write(f"  Mean: [{np.mean(end_points[:, 0]):.6f}, {np.mean(end_points[:, 1]):.6f}, {np.mean(end_points[:, 2]):.6f}]\n")
                f.write(f"  Std:  [{np.std(end_points[:, 0]):.6f}, {np.std(end_points[:, 1]):.6f}, {np.std(end_points[:, 2]):.6f}]\n")
        
        print(f"Results summary saved to: {summary_path}")


def main():
    """主函数"""
    # 初始化泛化性测试器
    tester = GeneralizationTester(model_type="pi0")  # 可以改为 "memo-vla"
    
    # 运行泛化性测试
    tester.run_generalization_test(
        use_camera=False  # 设为False使用固定图像，True使用实时相机
    )
    
    print("\n=== Generalization Test Completed ===")
    print(f"Results and visualizations saved to: {tester.output_dir}")


if __name__ == "__main__":
    main()