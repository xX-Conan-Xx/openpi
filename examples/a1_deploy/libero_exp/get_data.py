import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

DATA_ROOT = '/home/yuquan002/data/pine_demos/stack_2cups'

class PineDemoDataset(Dataset):
    def __init__(self, root_dir):
        self.demo_dirs = sorted(glob.glob(os.path.join(root_dir, 'demo_*')))
        self.samples = []
        for demo_dir in self.demo_dirs:
            wrist_imgs = sorted(glob.glob(os.path.join(demo_dir, 'left_camera', 'wrist_frame_*.jpg')))
            agent_imgs = sorted(glob.glob(os.path.join(demo_dir, 'right_camera', 'scene_frame_*.jpg')))
            ee_pose_file = os.path.join(demo_dir, 'end_effector_pose_left_arm.txt')
            joint_states_file = os.path.join(demo_dir, 'joint_states_left_arm.txt')
            # Load pose and joint data
            ee_pose = np.loadtxt(ee_pose_file)[:, 1:]  # skip first column (time)
            joint_states = np.loadtxt(joint_states_file)[:, 1:]  # skip first column (time)
            # Check data consistency
            n = min(len(wrist_imgs), len(agent_imgs), ee_pose.shape[0], joint_states.shape[0])
            for i in range(n):
                self.samples.append({
                    'wrist_img': wrist_imgs[i],
                    'agent_img': agent_imgs[i],
                    'ee_pose': ee_pose[i],
                    'joint_states': joint_states[i]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wrist_img = Image.open(sample['wrist_img']).convert('RGB')
        agent_img = Image.open(sample['agent_img']).convert('RGB')
        wrist_img = torch.from_numpy(np.array(wrist_img)).permute(2, 0, 1).float() / 255.0
        agent_img = torch.from_numpy(np.array(agent_img)).permute(2, 0, 1).float() / 255.0
        ee_pose = torch.tensor(sample['ee_pose'], dtype=torch.float32)
        joint_states = torch.tensor(sample['joint_states'], dtype=torch.float32)
        return {
            'wrist_img': wrist_img,
            'agent_img': agent_img,
            'ee_pose': ee_pose,
            'joint_states': joint_states
        }

# Print data info
demo_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, 'demo_*')))
print(f"Found {len(demo_dirs)} demos.")
for demo_dir in demo_dirs:
    wrist_imgs = sorted(glob.glob(os.path.join(demo_dir, 'left_camera', 'wrist_frame_*.jpg')))
    agent_imgs = sorted(glob.glob(os.path.join(demo_dir, 'right_camera', 'scene_frame_*.jpg')))
    ee_pose = np.loadtxt(os.path.join(demo_dir, 'end_effector_pose_left_arm.txt'))[:, 1:]
    joint_states = np.loadtxt(os.path.join(demo_dir, 'joint_states_left_arm.txt'))[:, 1:]
    print(f"{demo_dir}:")
    print(f"  wrist_imgs: {len(wrist_imgs)}, shape: {np.array(Image.open(wrist_imgs[0])).shape if wrist_imgs else 'N/A'}")
    print(f"  agent_imgs: {len(agent_imgs)}, shape: {np.array(Image.open(agent_imgs[0])).shape if agent_imgs else 'N/A'}")
    print(f"  ee_pose: {ee_pose.shape}")
    print(f"  joint_states: {joint_states.shape}")

# # Load with DataLoader
# dataset = PineDemoDataset(DATA_ROOT)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# # Example usage
# for batch in dataloader:
#     print("Batch wrist_img shape:", batch['wrist_img'].shape)
#     print("Batch agent_img shape:", batch['agent_img'].shape)
#     print("Batch ee_pose shape:", batch['ee_pose'].shape)
#     print("Batch joint_states shape:", batch['joint_states'].shape)
#     break

'''
(bcib) ➜  real_world_data python get_data.py
Found 240 demos.
/home/yuquan002/data/pine_demos/stack_2cups/demo_0:
  wrist_imgs: 173, shape: (480, 640, 3)
  agent_imgs: 173, shape: (480, 640, 3)
  ee_pose: (173, 7)
  joint_states: (173, 7)
'''