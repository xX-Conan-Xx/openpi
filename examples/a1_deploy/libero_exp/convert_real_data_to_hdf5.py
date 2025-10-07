# convert_real_data_to_hdf5.py

import os
import glob
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

def convert_quat_to_rpy(quat):
    '''
    Convert a quaternion to roll-pitch-yaw (RPY) angles.
    This is copied from your RealWorldDataset to ensure consistency.
    '''
    x, y, z, w = quat
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def convert_real_data_to_hdf5(root_dir, output_path, image_size=(128, 128)):
    """
    Scans a directory of real-world demos, processes them, and saves them
    into a single HDF5 file compatible with robomimic's SequenceDataset.

    Args:
        root_dir (str): The root directory containing demo folders (e.g., 'demo_0').
        output_path (str): Path to save the output HDF5 file.
        image_size (tuple): The (H, W) to which images should be resized.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    demo_dirs = sorted(glob.glob(os.path.join(root_dir, 'demo_*')))
    print(f"Found {len(demo_dirs)} demonstration folders in '{root_dir}'.")

    with h5py.File(output_path, 'w') as f:
        data_grp = f.create_group("data")
        total_samples = 0

        for demo_dir in tqdm(demo_dirs, desc="Processing demos"):
            demo_id = os.path.basename(demo_dir)
            try:
                # --- 1. Load all data streams from files ---
                wrist_img_paths = sorted(glob.glob(os.path.join(demo_dir, 'left_camera', 'wrist_frame_*.jpg')))
                agent_img_paths = sorted(glob.glob(os.path.join(demo_dir, 'right_camera', 'scene_frame_*.jpg')))
                joint_states = np.loadtxt(os.path.join(demo_dir, 'joint_states_right_arm.txt'))[:, 1:]
                ee_pose = np.loadtxt(os.path.join(demo_dir, 'end_effector_pose_right_arm.txt'))[:, 1:]

                # --- 2. Align data and calculate actions ---
                num_steps = min(len(wrist_img_paths), len(agent_img_paths), len(joint_states), len(ee_pose))
                if num_steps < 2:
                    print(f"Skipping short demo '{demo_id}' with {num_steps} steps.")
                    continue

                # Action calculation (copied from your dataset)
                xyz = ee_pose[:num_steps, :3]
                quat = ee_pose[:num_steps, 3:]
                rpy = np.array([convert_quat_to_rpy(q) for q in quat])
                gripper = joint_states[:num_steps, 6:]

                # Prepend is used to make the delta calculation result in an array of length num_steps
                # For HDF5, we need actions of length num_steps - 1
                actions_xyz = np.diff(xyz, axis=0)
                actions_rpy = np.diff(rpy, axis=0)
                actions_gripper = np.diff(gripper, axis=0)
                actions = np.concatenate([actions_xyz, actions_rpy, actions_gripper], axis=1)
                
                # --- 3. Process images ---
                # Robomimic expects observations to have length num_steps
                agent_images = []
                for p in agent_img_paths[:num_steps]:
                    img = Image.open(p).convert('RGB').resize((image_size[1], image_size[0]))
                    agent_images.append(np.array(img))
                agent_images = np.array(agent_images, dtype=np.uint8)

                wrist_images = []
                for p in wrist_img_paths[:num_steps]:
                    img = Image.open(p).convert('RGB').resize((image_size[1], image_size[0]))
                    wrist_images.append(np.array(img))
                wrist_images = np.array(wrist_images, dtype=np.uint8)

                # --- 4. Create HDF5 group for this demonstration ---
                demo_grp = data_grp.create_group(demo_id)
                
                # --- 5. Store data using YOUR keys ---
                # Observations (length num_steps)
                obs_grp = demo_grp.create_group("obs")
                obs_grp.create_dataset("agentview_rgb", data=agent_images, compression="gzip")
                obs_grp.create_dataset("eye_in_hand_rgb", data=wrist_images, compression="gzip")
                obs_grp.create_dataset("joint_states", data=joint_states[:num_steps, :6].astype(np.float32), compression="gzip")
                obs_grp.create_dataset("gripper_states", data=joint_states[:num_steps, 6:].astype(np.float32), compression="gzip")
                
                # Actions (length num_steps - 1)
                demo_grp.create_dataset("actions", data=actions.astype(np.float32), compression="gzip")
                
                # Robomimic requires `num_samples` to be the number of transitions (i.e., actions)
                num_transitions = num_steps - 1
                demo_grp.attrs["num_samples"] = num_transitions
                total_samples += num_transitions

            except Exception as e:
                print(f"Failed to process demo '{demo_id}': {e}")
        
        data_grp.attrs["total"] = total_samples
        print(f"\nSuccessfully created HDF5 file at: {output_path}")
        print(f"Total samples (transitions) stored: {total_samples}")

def extract_keys_from_hdf5(hdf5_file_path):
    keys = []
    with h5py.File(hdf5_file_path, "r") as f:
        # Navigate through the groups and datasets to extract keys
        for demo_id in f.keys():
            demo_grp = f[demo_id]
            obs_grp = demo_grp["obs"]
            keys.extend(list(obs_grp.keys()))
    print(f"Extracted observation keys: {keys}")

if __name__ == '__main__':
    # --- Step 1: Set the root directory of your raw real-world data ---
    # This is where your 'demo_0', 'demo_1', etc. folders are located.
    # IMPORTANT: Please verify this path is correct.
    # raw_data_root = os.path.expanduser("~/workspace/real_world_data/stack_cups")
    raw_data_root = os.path.expanduser("/home/yuquan002/data/stack_cube")

    # --- Step 2: Set the desired output path for the HDF5 file ---
    # This will be the new file you use for training.
    # We'll place it in the same directory for convenience.
    output_hdf5_path = os.path.join("/home/yuquan002/workspace/real_world_data/stack_cube", "dataset.hdf5")

    # --- Step 3: Run the conversion ---
    if not os.path.isdir(raw_data_root):
        os.mkdir(os.path.dirname(output_hdf5_path))
        # if hdf5 file exists under this path, delete
        if os.path.isfile(output_hdf5_path):
            os.remove(output_hdf5_path)
            print(f"Deleted existing file at {output_hdf5_path}")
            print("Recreating HDF5 file...")

    convert_real_data_to_hdf5(raw_data_root, output_hdf5_path, image_size=(128, 128))