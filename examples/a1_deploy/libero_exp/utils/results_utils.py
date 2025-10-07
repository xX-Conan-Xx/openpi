import os
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from einops import rearrange

from .data_utils import raw_obs_to_tensor_obs
from .video_utils import video_pad_time, rearrange_videos, render_done_to_boundary


@torch.no_grad()
def rollout(cfg, env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, video_writer=None, device=None):
    policy.eval()
    all_env_indices = []
    all_task_indices = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    all_env_descriptions = []

    for env_description, (env_idx, task_idx, env, init_states, task_emb) in env_dict.items():
        all_env_indices.append(env_idx)
        all_task_indices.append(task_idx)
        all_succ = []
        all_horizon = []
        vid = []

        if video_writer != None:
            final_video_path = os.path.join(video_writer.video_path, str(task_idx) + '.' + env_description.split('/')[-1])
            if os.path.exists(final_video_path):
                files = os.listdir(final_video_path)
                if files != []:
                    scuccess_files = [file for file in files if file.endswith("True.mp4")]
                    all_succ += [len(scuccess_files) / cfg.env.env_num]
                    all_horizon += [0]
                    vid += [0]

                    all_env_succ.append(np.array(all_succ).astype(np.float32))
                    all_env_horizon.append(all_horizon)
                    env_vid.append(vid)  # [(b, t, c, h, w)]

                    print(f"***{task_idx} {env_description.split('/')[-1]}*** has done!!!")
                    continue
            
        for num_env_rollout in range(num_env_rollouts):
            print(f"\nEnv: {env_idx}, Task_id: {task_idx}, Rollout: {num_env_rollout+1}/{num_env_rollouts} running...")
            env.reset()
            env.seed(cfg.train.seed)
            policy.reset()
            if video_writer != None:
                video_writer.reset()

            obs = env.set_init_state(init_states)

            # simulate the physics without any actions
            # action_dim = cfg.policy.policy_head.network_kwargs.output_size
            # dummy_actions = np.zeros((cfg.env.env_num, action_dim))
            # for _ in range(5):
            #     env.step(dummy_actions)

            dones = [False] * cfg.env.env_num
            num_success = 0
            episode_frames = []
            
            for step_i in tqdm(range(horizon)):
                data = raw_obs_to_tensor_obs(obs, task_emb, cfg, device)
                a = policy.get_action(cfg, data)
                obs, r, done, info = env.step(a)

                video_img = []
                for k in range(cfg.env.env_num):
                    dones[k] = dones[k] or done[k]
                    video_img.append(obs[k]["agentview_image"][::-1])
                video_img = np.stack(video_img, axis=0)
                frame = rearrange(video_img, "b h w c -> b c h w")
                frame = render_done_to_boundary(frame, dones)
                episode_frames.append(frame)

                if video_writer != None:
                    video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")
            
                if all(dones):
                    break

            if video_writer != None:
                video_writer.get_last_info(num_env_rollout, dones, env_description, task_idx)
                video_writer.save()

            for k in range(cfg.env.env_num):
                num_success += int(dones[k])
            all_succ += [num_success / cfg.env.env_num]
            all_horizon += [step_i+1]

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        all_env_descriptions.append(env_description)
    
    results = {}
    for idx, (env_idx, task_idx) in enumerate(zip(all_env_indices, all_task_indices)):
        results[f"rollout/horizon_env{env_idx}_task{task_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}_task{task_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}_task{task_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}_task{task_idx}"] = env_vid[idx]

    return results


def merge_results(results: List[dict], compute_avg=True):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v

    if compute_avg:
        merged_results["rollout/horizon_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/horizon_env" in k]).flatten())
        merged_results["rollout/success_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/success_env" in k]).flatten())
        
    return merged_results


def save_success_rate(epoch, success_metrics, summary_file_path):
    success_metrics = {k.replace("rollout/", ""): v for k, v in success_metrics.items()}
    success_heads = list(success_metrics.keys())
    success_heads.remove("success_env_avg")
    success_heads = sorted(success_heads, key=lambda x: int(x.split("success_env")[-1].split('task')[-1]))
    success_heads.append("success_env_avg")
    success_heads = ["epoch"] + success_heads

    success_metrics["epoch"] = str(epoch)
    df = pd.DataFrame(success_metrics, index=[0])

    if os.path.exists(summary_file_path):
        old_summary = pd.read_csv(summary_file_path)
        df = pd.concat([df, old_summary], ignore_index=True)

    df = df[success_heads]
    df.to_csv(summary_file_path)
    