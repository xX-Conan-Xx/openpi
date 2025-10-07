import collections
import dataclasses
import logging
import math
import pathlib

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from PIL import Image
from pyrealsense_image import initialize_camera, stop_camera, get_L515_image  # get_frame
from openpi.examples.a1_deploy.misc.eef_control import execute_eef_IK
import jax.numpy as jnp
import jax
import time


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    replay_images = []


    default_prompt = "Pick green cube and put into the bowl."  
    # "Press the button of sanitizer."  "Stack green cube on blue cube."  "Pick green cube and put into the bowl."

    
    pipeline = initialize_camera()
    print("Camera pipeline initialized.")
    for i in range(50):
        image = get_L515_image(pipeline) 
        if image is not None:
            image.save("captured_image.png") # BGR here, trained with 
    loop_interval = 0.1  # 控制频率
    
    # single_action = (0.22, 0.123, 0.3, -0.035, 1, -0.0024, -0.00445, 1) # (0.333, 0.069, 0.24, 0.0049, -0.017, 0.9968, 0.0768, 1)
    pick_init_action = (0.333, 0.069, 0.24, 0.00, 0.00, 1, 0, 1)
    stack_init_action = (0.333, 0.069, 0.17, 0.00, 0.00, 1, 0, 1)
    press_init_action = (0.333, 0.069, 0.34, 0.00, 0.00, 1, 0, 1)
    execute_eef_IK(pick_init_action)
    
    from scipy.spatial.transform import Rotation as R
    def quaternion_to_rpy(quaternion):
        """
        将四元数转换为RPY欧拉角（roll, pitch, yaw）。

        :param quaternion: 包含四个元素的列表或元组，表示四元数 (x, y, z, w)。
        :return: 包含三个元素的列表，表示RPY欧拉角 (roll, pitch, yaw)，单位为弧度。
        """
        r = R.from_quat(quaternion)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return [roll, pitch, yaw]
    
    init_pose = pick_init_action[0:3]  # 初始位置
    init_quat = pick_init_action[3:7]  # 初始四元数
    init_rpy = quaternion_to_rpy(init_quat) 
    
    print("!!!Initial rpy:", init_rpy)

    step=0

    try:
        while step < 60: # 60 for pick, 80 for stack, 50 for press
        # while True:
            print("Starting new iteration of the control loop...")
            start_time = time.time()
            
            # 获取图像 384 384
            try:
                print("Attempting to get image from camera...")
                # image = get_frame(pipeline, target_width=384, target_height=384)
                image = get_L515_image(pipeline) 
                if image is not None:
                    # Convert the image to BGR format before saving
                    bgr_image = image.convert("RGB")  # Ensure it's in RGB first
                    bgr_image = bgr_image.split()[::-1]  # Reverse channels to BGR
                    bgr_image = Image.merge("RGB", bgr_image)
                    bgr_image.save(f"/home/luka/Wenkai/visualization/captured_image_{step}.png")
                if image is None:
                    print("No image captured from camera.")
                    continue
                print("Image acquired successfully.")
            except Exception as e:
                print(f"Failed to get image from camera: {e}")
                time.sleep(loop_interval)
                continue
            
            img_array = np.asarray(bgr_image).astype(np.uint8)[None, ...]
            # print("img_array shape:", img_array.shape)
            # obs = {
            #     "image": {
            #         "base_0_rgb": img_array
            #     },
            #     "image_mask": {
            #         "base_0_rgb": jnp.ones((1,), dtype=bool)
            #     },
            #     "state": jnp.zeros((1, 7), dtype=jnp.float32),  # dummy state
            #     "prompt": "Pick green cube and put into the bowl."
            # }
            

            # img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_array, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_array, args.resize_size, args.resize_size)
            )

            # # Save preprocessed image for replay video
            # replay_images.append(img)
            print("img shape:", img.shape)
            
            element = {
                "observation/image": img[0],
                "observation/wrist_image": wrist_img[0],
                "observation/state": np.zeros((7), dtype=np.float32),
                "prompt": "Pick green cube and put into the bowl.",
            }
            # print(element)
            action_chunk = client.infer(element)["actions"]
            
            print(action_chunk)
    except Exception as e:
        logging.error(f"Caught exception: {e}")
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)


#     # Start evaluation
#     total_episodes, total_successes = 0, 0
#     for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
#         # Get task
#         task = task_suite.get_task(task_id)

#         # Get default LIBERO initial states
#         initial_states = task_suite.get_task_init_states(task_id)

#         # Initialize LIBERO environment and task description
#         env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

#         # Start episodes
#         task_episodes, task_successes = 0, 0
#         for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
#             logging.info(f"\nTask: {task_description}")

#             # Reset environment
#             env.reset()
#             action_plan = collections.deque()

#             # Set initial states
#             obs = env.set_init_state(initial_states[episode_idx])

#             # Setup
#             t = 0
#             replay_images = []

#             logging.info(f"Starting episode {task_episodes+1}...")
#             while t < max_steps + args.num_steps_wait:
#                 try:
#                     # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
#                     # and we need to wait for them to fall
#                     if t < args.num_steps_wait:
#                         obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
#                         t += 1
#                         continue

#                     # Get preprocessed image
#                     # IMPORTANT: rotate 180 degrees to match train preprocessing
#                     img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
#                     wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
#                     img = image_tools.convert_to_uint8(
#                         image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
#                     )
#                     wrist_img = image_tools.convert_to_uint8(
#                         image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
#                     )

#                     # Save preprocessed image for replay video
#                     replay_images.append(img)

#                     if not action_plan:
#                         # Finished executing previous action chunk -- compute new chunk
#                         # Prepare observations dict
#                         element = {
#                             "observation/image": img,
#                             "observation/wrist_image": wrist_img,
#                             "observation/state": np.concatenate(
#                                 (
#                                     obs["robot0_eef_pos"],
#                                     _quat2axisangle(obs["robot0_eef_quat"]),
#                                     obs["robot0_gripper_qpos"],
#                                 )
#                             ),
#                             "prompt": str(task_description),
#                         }

#                         # Query model to get action
#                         action_chunk = client.infer(element)["actions"]
#                         assert (
#                             len(action_chunk) >= args.replan_steps
#                         ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
#                         action_plan.extend(action_chunk[: args.replan_steps])

#                     action = action_plan.popleft()

#                     # Execute action in environment
#                     obs, reward, done, info = env.step(action.tolist())
#                     if done:
#                         task_successes += 1
#                         total_successes += 1
#                         break
#                     t += 1

#                 except Exception as e:
#                     logging.error(f"Caught exception: {e}")
#                     break

#             task_episodes += 1
#             total_episodes += 1

#             # Save a replay video of the episode
#             suffix = "success" if done else "failure"
#             task_segment = task_description.replace(" ", "_")
#             imageio.mimwrite(
#                 pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
#                 [np.asarray(x) for x in replay_images],
#                 fps=10,
#             )

#             # Log current results
#             logging.info(f"Success: {done}")
#             logging.info(f"# episodes completed so far: {total_episodes}")
#             logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

#         # Log final results
#         logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
#         logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

#     logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
#     logging.info(f"Total episodes: {total_episodes}")


# def _get_libero_env(task, resolution, seed):
#     """Initializes and returns the LIBERO environment, along with the task description."""
#     task_description = task.language
#     task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
#     env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
#     env = OffScreenRenderEnv(**env_args)
#     env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
#     return env, task_description


# def _quat2axisangle(quat):
#     """
#     Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
#     """
#     # clip quaternion
#     if quat[3] > 1.0:
#         quat[3] = 1.0
#     elif quat[3] < -1.0:
#         quat[3] = -1.0

#     den = np.sqrt(1.0 - quat[3] * quat[3])
#     if math.isclose(den, 0.0):
#         # This is (close to) a zero degree rotation, immediately return
#         return np.zeros(3)

#     return (quat[:3] * 2.0 * math.acos(quat[3])) / den


