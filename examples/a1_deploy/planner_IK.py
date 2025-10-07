import time
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
    
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

def solve_ik_for_urdf(urdf_file, base_link, ee_link, target_position, target_orientation):
    # 设置张量和设备参数
    tensor_args = TensorDeviceType()

    start_time = time.time()
    # 从指定 URDF 构建机器人配置
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    print(f"🌟 机器人配置耗时: {(time.time() - start_time) * 1000:.2f} ms")
    
    start_time = time.time()
    # 构建 IK 求解器配置（此处未加载 world config，如需碰撞检测请传入对应的 world 配置）
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,  # 无 world config
        rotation_threshold=0.08,    # 旋转误差阈值
        position_threshold=0.01,   # 位置误差阈值
        num_seeds=10,               # 采样初始解个数
        self_collision_check=False, # 不进行自碰撞检测
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    print(f"🌟 IK 求解器配置耗时: {(time.time() - start_time) * 1000:.2f} ms")
    
    start_time = time.time()
    # 创建 IK 求解器
    ik_solver = IKSolver(ik_config)

    # 采样一批随机初始关节配置（用于 IK 优化初始猜测）
    # q_sample = ik_solver.sample_configs(20)
    print(f"🌟 采样初始关节配置耗时: {(time.time() - start_time) * 1000:.2f} ms")

    
    print("target position:", target_position)

    # 使用给定的目标末端位姿构造目标 Pose
    # 转换为 Python 浮点数列表
    target_position_list = list(target_position)
    target_position_tensor = torch.tensor(target_position_list, device=tensor_args.device, dtype=torch.float32)
    
    target_orientation_list = list(target_orientation)
    target_orientation_tensor = torch.tensor(target_orientation_list, device=tensor_args.device, dtype=torch.float32)
    # target_position_tensor = torch.tensor(target_position, device=tensor_args.device, dtype=torch.float32)
    # target_orientation_tensor = torch.tensor(target_orientation, device=tensor_args.device, dtype=torch.float32)
    goal = Pose(target_position_tensor, target_orientation_tensor)

    # 执行 IK 求解
    start_time = time.time()
    result = ik_solver.solve_batch(goal)
    # print("goal:",goal)
    elapsed_time = (time.time() - start_time) * 1000  # 转换成毫秒
    # print(f"🌟 IK 求解耗时: {elapsed_time:.2f} ms")
    torch.cuda.synchronize()

    # 输出求解结果及相关指标
    # success_rate = torch.count_nonzero(result.success).item() / len(q_sample)
    # solve_time = result.solve_time
    # throughput = q_sample.shape[0] / (time.time() - st_time)
    # pos_error = torch.mean(result.position_error)
    # rot_error = torch.mean(result.rotation_error)

    # print("成功率:", success_rate)
    # print("求解时间 (s):", solve_time)
    # print("吞吐率 (hz):", throughput)
    # print("平均位置误差:", pos_error)
    # print("平均旋转误差:", rot_error)
    print("Solved joint pose:", result.js_solution.position.cpu().numpy().flatten())

    return result

def main():
    # 求解关节角度解
    joint_solution = solve_ik_for_urdf(
        urdf_file = "/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf", 
        base_link = "base_link", 
        ee_link = "arm_seg6", 
        target_position = [0.2, 0.0, 0.5], 
        target_orientation = [0.5, 0.5, 0.5, 0.5])
    
    if joint_solution is not None:
        print("Solved joint pose:")
        print(joint_solution)
    else:
        print("!! Can not Solved joint pose")
        
if __name__ == "__main__":
    main()



'''
position: [0.06799983978271484, 0.7869997024536133, -0.7869997024536133, -0.04500007629394531, -0.07600021362304688, 0.06299972534179688]
'''