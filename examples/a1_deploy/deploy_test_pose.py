"""
PI0 VLA模型位姿泛化性测试脚本 - Pose Generalization Test
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

class PoseGeneralizationTester:
    
    def __init__(self, model_type="pi0", websocket_host="0.0.0.0", websocket_port=8000):
        self.model_type = model_type.lower()
        self.pipelines = None
        
        # 初始化模型客户端
        if self.model_type in ["pi0", "memo-vla"]:
            self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
            print(f"Initialized {self.model_type.upper()} client at {websocket_host}:{websocket_port}")
        
        # 设置默认状态
        self.default_state = np.array([0.3959, -0.0129, 0.2078, 3.1416, 0.0000, 3.1416, -1.0000])
        
        # 固定测试指令
        self.test_prompt = "Place the bread on the plate."
        
        # 定义位姿变化（XYZ各增减0.05）
        self.pose_variations = self.generate_pose_variations()
        
        # 存储结果
        self.results = {}
        
        # 创建输出目录
        self.output_dir = "/home/luka/Wenkai/figs/generalization_test"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_pose_variations(self):
        """生成位姿变化组合"""
        variations = {}
        offset = 0.05
        
        # 原始位姿
        variations["Default"] = self.default_state.copy()
        
        # X坐标变化
        x_plus = self.default_state.copy()
        x_plus[0] += offset
        variations["X+0.05"] = x_plus
        
        x_minus = self.default_state.copy()
        x_minus[0] -= offset
        variations["X-0.05"] = x_minus
        
        # Y坐标变化
        y_plus = self.default_state.copy()
        y_plus[1] += offset
        variations["Y+0.05"] = y_plus
        
        y_minus = self.default_state.copy()
        y_minus[1] -= offset
        variations["Y-0.05"] = y_minus
        
        # Z坐标变化
        z_plus = self.default_state.copy()
        z_plus[2] += offset
        variations["Z+0.05"] = z_plus
        
        z_minus = self.default_state.copy()
        z_minus[2] -= offset
        variations["Z-0.05"] = z_minus
        
        return variations
    
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
    
    def prepare_model_input(self, main_image, wrist_image, state, prompt):
        """准备模型输入数据"""
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
    
    def extract_xyz_actions(self, result, input_state):
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
                # 每一行都加上输入状态的前3个值
                xyz_sequence_with_offset = xyz_sequence + input_state[:3]
                print(f"Extracted XYZ sequence shape: {xyz_sequence_with_offset.shape}")
                print(f"Input state XYZ: {input_state[:3]}")
                print(f"First output XYZ: {xyz_sequence_with_offset[0]}")
                print(f"Last output XYZ: {xyz_sequence_with_offset[-1]}")
                return xyz_sequence_with_offset
            elif actions_array.ndim == 1 and len(actions_array) >= 3:
                # 如果是一维数组，取前3个
                xyz_actions = actions_array[:3].reshape(1, -1) + input_state[:3]
                print(f"Extracted XYZ from 1D array: {xyz_actions}")
                return xyz_actions
            else:
                print(f"Unexpected actions format: shape {actions_array.shape}")
                return None
        
        return None
    
    def run_pose_generalization_test(self, use_camera=False):
        """运行位姿泛化性测试"""
        print("=== PI0 VLA Pose Generalization Test ===")
        print(f"Testing prompt: '{self.test_prompt}'")
        
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
        
        print(f"\nTesting {len(self.pose_variations)} different pose variations:")
        print("=" * 70)
        
        # 对每个位姿变化进行测试
        for i, (pose_label, state) in enumerate(self.pose_variations.items(), 1):
            print(f"\n[Test {i}/{len(self.pose_variations)}] Pose: {pose_label}")
            print(f"State XYZ: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}]")
            print("-" * 50)
            
            # 准备模型输入
            element = self.prepare_model_input(main_image, wrist_image, state, self.test_prompt)
            
            # 测试模型推理
            result = self.test_model_inference(element)
            
            # 提取XYZ动作
            xyz_actions = self.extract_xyz_actions(result, state)
            
            if xyz_actions is not None:
                self.results[pose_label] = xyz_actions
                print(f"XYZ sequence shape: {xyz_actions.shape}")
                print(f"First XYZ: [{xyz_actions[0, 0]:.4f}, {xyz_actions[0, 1]:.4f}, {xyz_actions[0, 2]:.4f}]")
                print(f"Last XYZ:  [{xyz_actions[-1, 0]:.4f}, {xyz_actions[-1, 1]:.4f}, {xyz_actions[-1, 2]:.4f}]")
            else:
                print("Failed to extract XYZ actions")
                self.results[pose_label] = None
        
        print("\n" + "=" * 70)
        print("All pose tests completed!")
        
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
        
        # 为每个位姿变化绘制轨迹
        colors = plt.cm.Set1(np.linspace(0, 1, len(valid_results)))
        
        for i, (pose_label, xyz_sequence) in enumerate(valid_results.items()):
            # xyz_sequence 是 [50, 3] 的形状
            x_coords = xyz_sequence[:, 0]
            y_coords = xyz_sequence[:, 1] 
            z_coords = xyz_sequence[:, 2]
            
            # 根据位姿类型设置不同的线型
            if pose_label == "Default":
                linestyle = '-'
                linewidth = 3.0
                alpha = 1.0
            else:
                linestyle = '-'
                linewidth = 2.5
                alpha = 0.8
            
            # 绘制轨迹线
            ax.plot(x_coords, y_coords, z_coords, 
                   color=colors[i], alpha=alpha, linewidth=linewidth, 
                   linestyle=linestyle, label=pose_label)
            
            # 标记起始点和结束点
            start_scatter = ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                                     color=colors[i], s=150, marker='o', alpha=0.9, 
                                     edgecolors='black', linewidths=1)
            end_scatter = ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                                   color=colors[i], s=150, marker='s', alpha=0.9, 
                                   edgecolors='black', linewidths=1)
            
            # 为默认位姿添加特殊标注
            if pose_label == "Default":
                ax.text(x_coords[0], y_coords[0], z_coords[0], '  begin(default)', 
                       fontsize=10, color='darkred', fontweight='bold')
                ax.text(x_coords[-1], y_coords[-1], z_coords[-1], '  end(default)', 
                       fontsize=10, color='darkblue', fontweight='bold')
            else:
                ax.text(x_coords[0], y_coords[0], z_coords[0], '  begin', 
                       fontsize=9, color='darkred', fontweight='bold')
                ax.text(x_coords[-1], y_coords[-1], z_coords[-1], '  end', 
                       fontsize=9, color='darkblue', fontweight='bold')
        
        # 设置轴标签
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_zlabel('Z Position (m)', fontsize=14)
        
        # 设置标题
        ax.set_title('PI0 VLA Model Pose Generalization Test\nTrajectories for Different Initial Poses\nPrompt: "Place the bread on the plate."', 
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
        output_path = os.path.join(self.output_dir, 'pose_generalization_test_3d_trajectories.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D pose trajectory visualization saved to: {output_path}")
        
        # 创建2D投影图
        self.create_2d_projections(valid_results)
        
        plt.show()
    
    def create_2d_projections(self, valid_results):
        """创建2D投影图"""
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(valid_results)))
        
        for i, (pose_label, xyz_sequence) in enumerate(valid_results.items()):
            x_coords = xyz_sequence[:, 0]
            y_coords = xyz_sequence[:, 1]
            z_coords = xyz_sequence[:, 2]
            
            # 根据位姿类型设置不同的线型
            if pose_label == "Default":
                linewidth = 3.0
                alpha = 1.0
            else:
                linewidth = 2.5
                alpha = 0.8
            
            # XY平面投影
            axes[0].plot(x_coords, y_coords, color=colors[i], alpha=alpha, linewidth=linewidth, label=pose_label)
            axes[0].scatter(x_coords[0], y_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[0].scatter(x_coords[-1], y_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            # 添加begin/end标注（仅为默认位姿）
            if pose_label == "Default":
                axes[0].annotate('begin', (x_coords[0], y_coords[0]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkred', fontweight='bold')
                axes[0].annotate('end', (x_coords[-1], y_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkblue', fontweight='bold')
            axes[0].set_xlabel('X Position (m)', fontsize=12)
            axes[0].set_ylabel('Y Position (m)', fontsize=12)
            axes[0].set_title('XY Plane Projection', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # XZ平面投影
            axes[1].plot(x_coords, z_coords, color=colors[i], alpha=alpha, linewidth=linewidth, label=pose_label)
            axes[1].scatter(x_coords[0], z_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[1].scatter(x_coords[-1], z_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            if pose_label == "Default":
                axes[1].annotate('begin', (x_coords[0], z_coords[0]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkred', fontweight='bold')
                axes[1].annotate('end', (x_coords[-1], z_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkblue', fontweight='bold')
            axes[1].set_xlabel('X Position (m)', fontsize=12)
            axes[1].set_ylabel('Z Position (m)', fontsize=12)
            axes[1].set_title('XZ Plane Projection', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # YZ平面投影
            axes[2].plot(y_coords, z_coords, color=colors[i], alpha=alpha, linewidth=linewidth, label=pose_label)
            axes[2].scatter(y_coords[0], z_coords[0], color=colors[i], s=120, marker='o', alpha=0.9, edgecolors='black', linewidths=1)
            axes[2].scatter(y_coords[-1], z_coords[-1], color=colors[i], s=120, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
            if pose_label == "Default":
                axes[2].annotate('begin', (y_coords[0], z_coords[0]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkred', fontweight='bold')
                axes[2].annotate('end', (y_coords[-1], z_coords[-1]), xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, color='darkblue', fontweight='bold')
            axes[2].set_xlabel('Y Position (m)', fontsize=12)
            axes[2].set_ylabel('Z Position (m)', fontsize=12)
            axes[2].set_title('YZ Plane Projection', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        
        # 在图形下方添加大的图例
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12, 
                  frameon=True, fancybox=True, shadow=True)
        
        # 保存2D投影图
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
        output_path_2d = os.path.join(self.output_dir, 'pose_generalization_test_2d_projections.png')
        plt.savefig(output_path_2d, dpi=300, bbox_inches='tight')
        print(f"2D pose trajectory projections saved to: {output_path_2d}")
    
    def save_results_summary(self):
        """保存结果摘要"""
        summary_path = os.path.join(self.output_dir, 'pose_generalization_test_results.txt')
        
        with open(summary_path, 'w') as f:
            f.write("PI0 VLA Model Pose Generalization Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Type: {self.model_type.upper()}\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Prompt: '{self.test_prompt}'\n")
            f.write(f"Default State: {self.default_state}\n\n")
            
            f.write("Pose Variations Tested:\n")
            f.write("-" * 30 + "\n")
            for pose_label, state in self.pose_variations.items():
                f.write(f"{pose_label}: XYZ=[{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}]\n")
            f.write("\n")
            
            f.write("Results for each pose variation:\n")
            f.write("-" * 40 + "\n")
            
            for i, (pose_label, xyz_sequence) in enumerate(self.results.items(), 1):
                f.write(f"{i}. Pose: {pose_label}\n")
                if xyz_sequence is not None:
                    f.write(f"   Input XYZ: {self.pose_variations[pose_label][:3]}\n")
                    f.write(f"   Trajectory shape: {xyz_sequence.shape}\n")
                    f.write(f"   Start XYZ: [{xyz_sequence[0, 0]:.6f}, {xyz_sequence[0, 1]:.6f}, {xyz_sequence[0, 2]:.6f}]\n")
                    f.write(f"   End XYZ:   [{xyz_sequence[-1, 0]:.6f}, {xyz_sequence[-1, 1]:.6f}, {xyz_sequence[-1, 2]:.6f}]\n")
                    
                    # 计算与默认位姿的差异
                    if pose_label != "Default" and "Default" in self.results and self.results["Default"] is not None:
                        default_trajectory = self.results["Default"]
                        start_diff = xyz_sequence[0] - default_trajectory[0]
                        end_diff = xyz_sequence[-1] - default_trajectory[-1]
                        f.write(f"   Start diff from default: [{start_diff[0]:.6f}, {start_diff[1]:.6f}, {start_diff[2]:.6f}]\n")
                        f.write(f"   End diff from default:   [{end_diff[0]:.6f}, {end_diff[1]:.6f}, {end_diff[2]:.6f}]\n")
                else:
                    f.write("   Trajectory: Failed to extract\n")
                f.write("\n")
            
            # 添加统计信息
            valid_results = {k: v for k, v in self.results.items() if v is not None}
            if valid_results and len(valid_results) > 1:
                # 统计所有轨迹的起始点和结束点
                start_points = np.array([xyz[0] for xyz in valid_results.values()])
                end_points = np.array([xyz[-1] for xyz in valid_results.values()])
                
                f.write("Statistical Summary:\n")
                f.write("-" * 20 + "\n")
                f.write("Start Points Across All Poses:\n")
                f.write(f"  Mean: [{np.mean(start_points[:, 0]):.6f}, {np.mean(start_points[:, 1]):.6f}, {np.mean(start_points[:, 2]):.6f}]\n")
                f.write(f"  Std:  [{np.std(start_points[:, 0]):.6f}, {np.std(start_points[:, 1]):.6f}, {np.std(start_points[:, 2]):.6f}]\n")
                f.write("End Points Across All Poses:\n")
                f.write(f"  Mean: [{np.mean(end_points[:, 0]):.6f}, {np.mean(end_points[:, 1]):.6f}, {np.mean(end_points[:, 2]):.6f}]\n")
                f.write(f"  Std:  [{np.std(end_points[:, 0]):.6f}, {np.std(end_points[:, 1]):.6f}, {np.std(end_points[:, 2]):.6f}]\n")
                
                # 计算轨迹一致性
                f.write("\nTrajectory Consistency Analysis:\n")
                f.write("-" * 35 + "\n")
                f.write("Input pose sensitivity (how much output changes with input changes):\n")
                
                if "Default" in valid_results:
                    default_start = valid_results["Default"][0]
                    default_end = valid_results["Default"][-1]
                    
                    for pose_label, trajectory in valid_results.items():
                        if pose_label != "Default":
                            input_diff = self.pose_variations[pose_label][:3] - self.pose_variations["Default"][:3]
                            output_start_diff = trajectory[0] - default_start
                            output_end_diff = trajectory[-1] - default_end
                            
                            # 计算敏感性比率
                            start_sensitivity = np.linalg.norm(output_start_diff) / np.linalg.norm(input_diff) if np.linalg.norm(input_diff) > 0 else 0
                            end_sensitivity = np.linalg.norm(output_end_diff) / np.linalg.norm(input_diff) if np.linalg.norm(input_diff) > 0 else 0
                            
                            f.write(f"  {pose_label}: Start sensitivity={start_sensitivity:.3f}, End sensitivity={end_sensitivity:.3f}\n")
        
        print(f"Pose generalization results summary saved to: {summary_path}")


def main():
    """主函数"""
    # 初始化位姿泛化性测试器
    tester = PoseGeneralizationTester(model_type="pi0")  # 可以改为 "memo-vla"
    
    # 运行位姿泛化性测试
    tester.run_pose_generalization_test(
        use_camera=False  # 设为False使用固定图像，True使用实时相机
    )
    
    print("\n=== Pose Generalization Test Completed ===")
    print(f"Results and visualizations saved to: {tester.output_dir}")


if __name__ == "__main__":
    main()