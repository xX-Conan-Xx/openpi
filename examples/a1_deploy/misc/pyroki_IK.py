"""
说明：
- 仅做一次“预热”触发 JAX/XLA 编译，后续 fast_ik_once 单点求解即可维持 30–70 ms。
- pks.solve_ik 内部已经是 JAX 实现，无需再套 @jax.jit。
"""

import time, os, sys
import numpy as np
import pyroki as pk
from yourdfpy import URDF
import jax
import jax.numpy as jnp

print("🖥️  设备:", jax.devices())

# ---------- 让 Python 能找到 pyroki_snippets ----------
PYROKI_SNIPPETS_PATH = "/home/luka/Wenkai/openpi/third_party/pyroki/examples"
if PYROKI_SNIPPETS_PATH not in sys.path:
    sys.path.append(PYROKI_SNIPPETS_PATH)

import pyroki_snippets as pks
print("✅ 已加载 pyroki_snippets")

# ---------- 预加载模型 ----------
URDF_PATH = "/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
assert os.path.exists(URDF_PATH), f"URDF 不存在: {URDF_PATH}"

urdf        = URDF.load(URDF_PATH)
TARGET_LINK = "arm_seg6"
robot       = pk.Robot.from_urdf(urdf)
print("✅ 机器人模型加载完成")

# ---------- 首次调用：编译预热 ----------
_dummy_pos  = np.array([0., 0., 0.],  dtype=np.float32)
_dummy_wxyz = np.array([0., 0., 0., 1.], dtype=np.float32)

print("🚀 首次调用（编译预热）…")
_ = pks.solve_ik(
        robot            = robot,
        target_link_name = TARGET_LINK,
        target_position  = _dummy_pos,
        target_wxyz      = _dummy_wxyz,
    )
print("✅ 预热完成，后续求解将更快")

# ---------- 快速单点 IK ----------
def fast_ik_once(position_xyz, orientation_wxyz):
    """
    单次 IK 求解（已预热）。
    典型 GPU 耗时 30–70 ms；CPU 约 200–300 ms（取决于硬件）。
    """
    start = time.time()
    sol   = pks.solve_ik(
        robot            = robot,
        target_link_name = TARGET_LINK,
        target_position  = np.array(position_xyz,  dtype=np.float32),
        target_wxyz      = np.array(orientation_wxyz, dtype=np.float32),
    )
    print(f"✅ IK 成功，耗时: {(time.time()-start)*1000:.1f} ms")
    print("   关节角度:", sol)
    return sol

# ---------- 使用示例 ----------
if __name__ == "__main__":
    target_pos  = [0.2, 0.0, 0.5]
    target_wxyz = [0.5, 0.5, 0.5, 0.5]
    fast_ik_once(target_pos, target_wxyz)
