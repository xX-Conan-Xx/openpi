#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ---------- math utils ----------
def quat_to_R_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x*s, y*s, z*s
    wx, wy, wz = w*X, w*Y, w*Z
    xx, xy, xz = x*X, x*Y, x*Z
    yy, yz, zz = y*Y, y*Z, z*Z
    return np.array([
        [1.0 - (yy + zz),     xy - wz,           xz + wy],
        [xy + wz,             1.0 - (xx + zz),   yz - wx],
        [xz - wy,             yz + wx,           1.0 - (xx + yy)],
    ], dtype=np.float32)

def R_to_rpy_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll  = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[0, 2], R[0, 1])
        yaw  = 0.0
    return roll, pitch, yaw

def read_end_effector_right(path_txt: Path):
    """
    File format per row:
      t, px, py, pz, qx, qy, qz, qw  (xyzw)
    Returns: times (N,), positions (N,3), quats_xyzw (N,4)
    """
    M = np.loadtxt(path_txt)
    t = M[:, 0]
    p = M[:, 1:4]
    q_xyzw = M[:, 4:8]  # already [qx, qy, qz, qw]
    # normalize just in case
    q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw, axis=1, keepdims=True) + 1e-12)
    return t, p, q_xyzw


def read_joint_states_right(path_txt: Path):
    M = np.loadtxt(path_txt)
    t = M[:, 0]
    gripper = M[:, -1]
    return t, gripper

def list_images_sorted(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

def load_and_resize_image(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(im, dtype=np.uint8)

# ---------- main conversion ----------
def build_state_from_pose_and_gripper(p_xyz: np.ndarray,
                                             q_xyzw: np.ndarray,
                                             gripper: np.ndarray) -> np.ndarray:
    N = min(len(p_xyz), len(q_xyzw), len(gripper))
    out = np.zeros((N, 7), dtype=np.float32)
    gripper_bool = (gripper < -2.0).astype(np.float32)  # cartin: -3.0 closed, -1.0 open
    for i in range(N):
        R = quat_to_R_xyzw(q_xyzw[i])
        r, p, y = R_to_rpy_xyz(R)
        out[i, 0:3] = p_xyz[i]
        out[i, 3:6] = [r, p, y]
        out[i, 6]   = gripper_bool[i]
    return out

def build_action_from_difference_of_states(states: np.ndarray) -> np.ndarray:
    N = len(states)
    actions = np.zeros((N, 7), dtype=np.float32)

    # set last action to zero
    actions[-1,0:6] = np.zeros((6,), dtype=np.float32)
    actions[-1,6] = states[-1,6]

    # keep the last column (gripper) the same as states
    # compute difference for the first 6 columns
    for i in range(N - 1):
        actions[i, 0:6] = states[i + 1, 0:6] - states[i, 0:6]

        if actions[i, 3] > math.pi:
            print("Huge jump identified in roll; correcting, roll:", actions[i, 3])
            actions[i,3] = math.pi *2 - actions[i,3]
            print("Corrected roll action:", actions[i, 3])
        if actions[i, 3] < -1 * math.pi:
            print("Huge jump identified in roll; correcting, roll:", actions[i, 3])
            actions[i,3] = math.pi *2 + actions[i,3]
            print("Corrected roll action:", actions[i, 3])

        
        if actions[i, 4] > math.pi:
            print("Huge jump identified in pitch; correcting, pitch:", actions[i, 4])
            actions[i,4] = math.pi *2 - actions[i,4]
            print("Corrected pitch action:", actions[i, 4])
        if actions[i, 4] < -1 * math.pi:
            print("Huge jump identified in pitch; correcting, pitch:", actions[i, 4])
            actions[i,4] = math.pi *2 + actions[i,4]
            print("Corrected pitch action:", actions[i, 4])
        

        if actions[i, 5] > math.pi:
            print("Huge jump identified in yaw; correcting, yaw:", actions[i, 5])
            actions[i,5] = math.pi *2 - actions[i,5]
            print("Corrected yaw action:", actions[i, 5])
        if actions[i, 5] < -1 * math.pi:
            print("Huge jump identified in yaw; correcting, yaw:", actions[i, 5])
            actions[i,5] = math.pi *2 + actions[i,5]
            print("Corrected yaw action:", actions[i, 5])

        
        actions[i, 6] = states[i, 6]

    return actions

def convert_episode(demo_dir: Path, dataset: LeRobotDataset, img_size: int, task_name: str):
    # signals
    t_pose, p_xyz, q_xyzw = read_end_effector_right(demo_dir / "end_effector_pose_right_arm.txt")
    t_joint, gripper = read_joint_states_right(demo_dir / "joint_states_right_arm.txt")

    N = min(len(t_pose), len(t_joint))
    p_xyz, q_xyzw, gripper = p_xyz[:N], q_xyzw[:N], gripper[:N]
    states = build_state_from_pose_and_gripper(p_xyz, q_xyzw, gripper)  # (N,7)
    actions = build_action_from_difference_of_states(states)  # (N,7)
    # cameras
    right_dir = demo_dir / "right_camera"
    left_dir  = demo_dir / "left_camera"
    right_imgs = list_images_sorted(right_dir)
    left_imgs  = list_images_sorted(left_dir) if left_dir.exists() else []

    if len(right_imgs) == 0:
        raise FileNotFoundError(f"No images in {right_dir}")

    M = min(len(right_imgs), len(left_imgs) if left_imgs else len(right_imgs), N)

    for i in range(M):
        frame = {
            "image": load_and_resize_image(right_imgs[i], img_size),   # RIGHT (kept as "image")
            "state": states[i],
            "actions": actions[i],
            "task": task_name,
        }
        if left_imgs:
            frame["wrist_image"] = load_and_resize_image(left_imgs[i], img_size)  # LEFT stream
        dataset.add_frame(frame)
    dataset.save_episode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--repo_id", type=str, required=True)
    ap.add_argument("--task_name", type=str, default="custom_task")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    demos = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("demo_")])
    if not demos:
        raise FileNotFoundError(f"No demo_* folders found under {data_root}")

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        import shutil; shutil.rmtree(out_path)

    # Add "left_image" feature
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="custom",
        fps=args.fps,
        features={
            "image": {        # right camera (unchanged)
                "dtype": "image",
                "shape": (args.img_size, args.img_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {   # left camera (new)
                "dtype": "image",
                "shape": (args.img_size, args.img_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x","y","z","roll","pitch","yaw","gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x","y","z","roll","pitch","yaw","gripper"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    for d in demos:
        need = ["end_effector_pose_right_arm.txt", "joint_states_right_arm.txt", "right_camera"]
        if any(not (d / n).exists() for n in need):
            print(f"[skip] missing files in {d.name}")
            continue
        print(f"[convert] {d.name}")
        convert_episode(d, dataset, args.img_size, args.task_name)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["lerobot","custom","right_arm",args.task_name],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    main()
