#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IK Solver for A1 arm using ikpy, wrapped into a class
"""
import sys
import os

# sys.path.insert(0, "/home/luka/Wenkai/openpi/third_party")
sys.path.append(os.path.join(os.path.dirname(__file__), "../..")) 

import numpy as np
from ikpy.src.ikpy.chain import Chain
from ikpy.src.ikpy.link import OriginLink
from scipy.spatial.transform import Rotation
import time

class IKSolver:
    def __init__(self, urdf_path, base_link, ee_link):
        """
        初始化 IK Solver，加载 URDF 文件并解析链结构
        """
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link

        # 1️⃣ 加载 URDF 构建链
        self.arm_chain = Chain.from_urdf_file(self.urdf_path)

        # 2️⃣ 查找末端执行器索引
        self.ee_index = None
        for idx, link in enumerate(self.arm_chain.links):
            if link.name == self.ee_link:
                self.ee_index = idx
                break

        if self.ee_index is None:
            raise ValueError(f"未找到末端执行器 {self.ee_link} 的链接，请检查 URDF 文件。")
        else:
            print(f"🔍 找到末端执行器 `{self.ee_link}`，在链中的索引是 {self.ee_index}")

        # 3️⃣ 手动截断链条到末端执行器
        self.arm_chain.links = self.arm_chain.links[:self.ee_index + 1]

        # 4️⃣ 修正 active_links_mask，去掉所有固定关节
        new_active_mask = []
        for link in self.arm_chain.links:
            if link.joint_type != "fixed":
                new_active_mask.append(True)
            else:
                new_active_mask.append(False)

        # 更新 active mask
        self.arm_chain.active_links_mask = new_active_mask
        print(f"✅ 修正后的 active_links_mask: {self.arm_chain.active_links_mask}")

    def solve_ik(self, xyz, quat_wxyz, orientation_mode="Z", **ik_kwargs):
        """
        逆运动学求解
        Parameters
        ----------
        xyz : list of float
            末端目标位置 [x, y, z]
        quat_wxyz : list of float
            目标末端姿态 [w, x, y, z]
        orientation_mode : str
            选择对齐的轴："X", "Y", "Z" 或 "all"
        ik_kwargs : dict
            额外传递给 ikpy 的求解参数

        Returns
        -------
        np.ndarray
            求解后的关节角度
        """
        target_pos = np.asarray(xyz, dtype=float)

        # 转换四元数到旋转矩阵
        rmat = Rotation.from_quat(quat_wxyz[1:] + quat_wxyz[:1]).as_matrix()

        # 选取对齐模式
        if orientation_mode == "X":
            target_ori = rmat[:, 0]
        elif orientation_mode == "Y":
            target_ori = rmat[:, 1]
        elif orientation_mode == "Z":
            target_ori = rmat[:, 2]
        elif orientation_mode == "all":
            target_ori = rmat
        else:
            raise ValueError(f"Unsupported orientation_mode '{orientation_mode}'")

        # 逆运动学求解
        start_time = time.time()
        thetas = self.arm_chain.inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_ori,
            orientation_mode=orientation_mode,
            **ik_kwargs
        )
        print(f"IK 求解耗时: {(time.time() - start_time) * 1000:.1f} ms")
        return thetas


# -------------------------------
# ✧✧✧ 测试 ✧✧✧
# -------------------------------
if __name__ == "__main__":
    solver = IKSolver(
        urdf_path="/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
        base_link="base_link",
        ee_link="arm_joint6"
    )
    joint_angles = solver.solve_ik(
        xyz=[0.2, 0.0, 0.5],
        quat_wxyz=[0.5, 0.5, 0.5, 0.5],
        orientation_mode="Z",
        max_iter=1000
    )
    print("IK 结果 (rad):", joint_angles)
