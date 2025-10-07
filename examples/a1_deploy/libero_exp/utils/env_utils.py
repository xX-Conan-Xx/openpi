import os
import time
import math
import torch
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv, SubprocVectorEnv

from .data_utils import get_task_embs


def build_env(cfg, env_type, env_name, task_id=None, img_size=128, 
               render_gpu_ids=-1, env_num=1, env_idx_start_end=None, **kwargs):
    """
    Build the rollout environment.
    Args:
        img_size: The resolution of the pixel observation.
        env_type: The type of environment benchmark. Choices: ["libero"].
        env_name: The name to specify the environments.
        render_gpu_ids: The available GPU ids for rendering the images
        env_num: The number of parallel environments
        seed: The random seed environment initialization.

    Returns:
        env: A gym-like environment.
    """
    if env_type.lower() == "libero":
        if isinstance(render_gpu_ids, Iterable):
            render_gpu_ids = [int(i) for i in render_gpu_ids]
            gpu_id_for_each_env = render_gpu_ids * math.ceil(len(env_name) / len(render_gpu_ids))
            gpu_id_for_each_env = gpu_id_for_each_env[:len(env_name)]
        else:
            gpu_id_for_each_env = [render_gpu_ids] * len(env_name)

        if env_idx_start_end is not None:
            idx_start, idx_end = env_idx_start_end
        else:
            idx_start = 0
            idx_end = len(env_name)

        env_dict = OrderedDict()
        for env_idx in range(idx_start, idx_end):
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[env_name[env_idx]]()

            if task_id == None:
                task_id = range(task_suite.n_tasks)

            task_descriptions = []
            for task_i in range(task_suite.n_tasks):
                task_descriptions.append(task_suite.get_task(task_i).language)

            embedding_model_path = cfg.data.embedding_model_path
            cwd = os.path.dirname(os.path.abspath(__file__))
            file_path = f"{cwd}/../data/{cfg.data.env_name}_task_embeddings.pt"
            if os.path.exists(file_path):
                task_embs = torch.load(file_path)
            else:
                task_embs = get_task_embs(cfg, task_descriptions, embedding_model_path)
                torch.save(task_embs, file_path)     
            task_suite.set_task_embs(task_embs)

            for task_i in task_id: 
                env, init_states_ = make_libero_env(task_suite, task_i, img_size, env_num, gpu_id_for_each_env[env_idx])
                task_emb = task_suite.get_task_emb(task_i)
                env_dict[f"{env_name[env_idx]}/{task_suite.get_task(task_i).name}"] = (env_idx, task_i, env, init_states_, task_emb)
    else:
        raise ValueError(f"Environment {env_type} is not supported!")
    
    return env_dict


def make_libero_env(task_suite, task_id, img_size, env_num=1, gpu_id=-1):
    """
    Build a LIBERO environment according to the task suite name and task name.
    """    
    if isinstance(img_size, Iterable):
        assert len(img_size) == 2
        img_h = img_size[0]
        img_w = img_size[1]
    else:
        img_h = img_w = img_size

    task = task_suite.get_task(task_id)

    # retrieve a specific task
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task.name} from suite {task_suite.name}, the " + \
          f"language instruction is {task.language}, and the bddl file is {task.bddl_file}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_h,
        "camera_widths": img_w,
        "render_gpu_device_id": gpu_id,
    }

    init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)
    assert len(init_states) >= env_num, "error: the number of initial states must be more than the number of envs"
    indices = np.arange(env_num) % init_states.shape[0]
    init_states_ = init_states[indices]

    env_created = False
    count = 0
    env = None
    while not env_created and count < 5:
        try:
            if env_num == 1:
                env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
            else:
                env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
            env_created = True
        except:
            time.sleep(5)
            count += 1
    if count >= 5:
        raise Exception("Failed to create environment")
    
    return env, init_states_
