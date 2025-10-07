import os
import gc
import math
import time
import datetime
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
import logging
from tqdm import tqdm
from lightning.fabric import Fabric
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import SequenceVLDataset

from ..data.get_dataset import get_dataset
from ..models import BCRNNPolicy, BCTransformerPolicy, BCViLTPolicy, BCMLPPolicy, BCDPPolicy
from ..utils.data_utils import get_task_embs
from ..utils.env_utils import build_env
from ..utils.train_utils import setup_optimizer, setup_lr_scheduler
from ..utils.video_utils import VideoWriter
from ..utils.results_utils import rollout, merge_results, save_success_rate
from ..utils.record_utils import init_wandb, MetricLogger, BestAvgLoss, AverageMeter, MetricMeter

from ..data.realworld_dataset import RealWorldDataset, get_real_dataset
import robomimic.utils.obs_utils as ObsUtils


REGISTERED_ALGOS = {}


def register_algo(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """Get the policy class from the registry."""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all algorithms that should not be registered here.
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class BaseAlgo(nn.Module, metaclass=AlgoMeta):
    def __init__(self, cfg, fabric: Fabric, inference=False, device='cuda'):
        super().__init__()

        self.fabric = fabric
        self.device = device

        if cfg.data.env_name == 'libero_10':
            cfg.env.max_steps = 1000

        if not inference:
            print("Inference mode")
            self.device = device
            cfg.train.device = device
            # devce = 'cuda'
            if device == 'cuda':
                self.fabric = Fabric(accelerator="cuda", devices=list(cfg.train.train_gpus), strategy="ddp")
                self.fabric.launch()

            load_path = cfg.eval.load_path
            self.eval_result_dir = os.path.join(load_path, "eval_results")
            os.makedirs(self.eval_result_dir, exist_ok=True)
            if cfg.eval.eval_all:
                model_list = ["model_final.ckpt", "model_10.ckpt", "model_20.ckpt", "model_30.ckpt", "model_40.ckpt"]
            else:
                model_list = ["model_final.ckpt"]
            self.ckp_paths_to_eval = [os.path.join(load_path, model_i) for model_i in model_list]
            self.summary_file_path = os.path.join(self.eval_result_dir, f"summary_{cfg.env.env_name}.csv")
            cfg.env.env_name = [cfg.env.env_name]
            
        else:   # train
            print("Training mode")
            shape_meta = self.build_dataloader(cfg)
            self.device = cfg.train.device
            self.fabric = Fabric(accelerator="cuda", devices=list(cfg.train.train_gpus), precision="bf16-mixed" if cfg.train.mix_precision else None, strategy="deepspeed")
            self.fabric.launch()
            self.build_model(cfg, shape_meta)

            if cfg.eval.enable_rollout:
                cfg.env.env_num, cfg.env.num_env_rollouts = 10, 10
                cfg.env.render_gpu_ids = cfg.env.render_gpu_ids[self.fabric.global_rank] if isinstance(cfg.env.render_gpu_ids, list) else cfg.env.render_gpu_ids
                cfg.env.env_name = [cfg.env.env_name]
                env_num_each_rank = math.ceil(len(cfg.env.env_name) / self.fabric.world_size)
                env_idx_start_end = (env_num_each_rank * self.fabric.global_rank, min(env_num_each_rank * (self.fabric.global_rank + 1), len(cfg.env.env_name)))
                print("\nInitialing val enviroment...")
                self.rollout_env = build_env(cfg, img_size=cfg.data.img_size, env_idx_start_end=env_idx_start_end, **cfg.env)

            self.fabric.barrier()
            self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
            self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
            # self.val_loader = self.fabric.setup_dataloaders(self.val_loader)

        self.cfg = cfg

        print(self.fabric)

    def build_dataloader(self, cfg):
        # if load real-world dataset
        if cfg.data.get("is_real_world", False):
            logging.info(">>> Building DataLoader for REAL-WORLD data <<<")

            obs_config = {"obs": cfg.data.obs.modality}
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_config)

            # --- 1. New RealWorldDataset ---
            # train_dataset = RealWorldDataset(
            #     root_dir=cfg.data.real_world_data_dir,
            #     obs_modality=cfg.data.obs.modality,
            #     obs_key_mapping=cfg.data.obs_key_mapping,
            #     seq_len=cfg.data.seq_len,
            #     frame_stack=cfg.data.frame_stack,
            #     image_size=(cfg.data.img_size, cfg.data.img_size)
            # )
            hdf5_path = os.path.expanduser("~/workspace/real_world_data/pick_pepper/dataset.hdf5")
            train_dataset, shape_meta = get_real_dataset(
                dataset_path=hdf5_path,
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.data.seq_len,
                frame_stack=cfg.data.frame_stack,
                train_ratio=cfg.data.train_ratio,
                train=True,
                val_demo_num=cfg.data.val_demo_num,
            )

            # train_dataset = RealWorldDataset(
            #     root_dir=cfg.data.real_world_data_dir,
            #     obs_modality=cfg.data.obs.modality,
            #     seq_len=cfg.data.seq_len,
            #     frame_stack=cfg.data.frame_stack,
            #     image_size=(cfg.data.img_size, cfg.data.img_size)
            # )


            # Create DataLoader for training
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers*2,
                # num_workers=1,
                shuffle=True,
                persistent_workers=True,
            )

            # Create an empty validation loader since we don't have val data for real-world training
            # val_dataset, _ = get_real_dataset(
            #     dataset_path=hdf5_path,
            #     obs_modality=cfg.data.obs.modality,
            #     seq_len=cfg.data.seq_len,
            #     frame_stack=cfg.data.frame_stack,
            #     train_ratio=cfg.data.train_ratio,
            #     train=False,
            #     val_demo_num=cfg.data.val_demo_num,
            # )
            # self.val_loader = DataLoader(
            #     val_dataset,
            #     batch_size=cfg.eval.batch_size,
            #     num_workers=4,
            #     shuffle=True,
            #     persistent_workers=True,
            # )

            # shape_meta = {
            #     "all_shapes": {
            #         "agentview_rgb": (cfg.data.img_height, cfg.data.img_width, 3),
            #         "eye_in_hand_rgb": (cfg.data.img_width, cfg.data.img_width, 3),
            #         "joint_states": (6,),
            #         "gripper_states": (1,), 
            #     },
            #     # action dims
            #     "ac_dim": 7
            # }

            # # observation keys
            # shape_meta["all_obs_keys"] = [k for k in shape_meta["all_shapes"].keys() if any(k in v for v in cfg.data.obs.modality.values())]

            return shape_meta

        # if not real-world dataset, use libero
        else:
            logging.info(">>> Building DataLoader for LIBERO data <<<")
            # load dataset
            benchmark = get_benchmark(cfg.data.env_name)(cfg.data.task_order_index)
            n_tasks = benchmark.n_tasks
            train_manip_datasets, val_manip_datasets = [], []
            descriptions = []

            for i in range(n_tasks):
                try:
                    task_i_train_dataset, shape_meta = get_dataset(
                        dataset_path=os.path.join(cfg.data.root_dir, benchmark.get_task_demonstration(i)),
                        obs_modality=cfg.data.obs.modality,
                        initialize_obs_utils=(i == 0),
                        seq_len=cfg.data.seq_len,
                        train_ratio=cfg.data.train_ratio,
                        train=True,
                        val_demo_num=cfg.data.val_demo_num,
                    )
                    task_i_val_dataset, _ = get_dataset(
                        dataset_path=os.path.join(cfg.data.root_dir, benchmark.get_task_demonstration(i)),
                        obs_modality=cfg.data.obs.modality,
                        initialize_obs_utils=False,
                        seq_len=cfg.data.seq_len,
                        train_ratio=cfg.data.train_ratio,
                        train=False,
                        val_demo_num=cfg.data.val_demo_num,
                    )
                except Exception as e:
                    print(f"[error] failed to load task {i}: {benchmark.get_task_names()[i]}")
                    print(f"[error] {e}")
                    raise
                
                train_manip_datasets.append(task_i_train_dataset)
                val_manip_datasets.append(task_i_val_dataset)
                task_description = benchmark.get_task(i).language
                descriptions.append(task_description)
                print(f"{i+1}. Loaded form {benchmark.get_task_demonstration(i)}.")

            embedding_model_path = cfg.data.embedding_model_path
            cwd = os.path.dirname(os.path.abspath(__file__))
            file_path = f"{cwd}/../data/{cfg.data.env_name}_task_embeddings.pt"
            if os.path.exists(file_path):
                task_embs = torch.load(file_path)
            else:
                task_embs = get_task_embs(cfg, descriptions, embedding_model_path)
                torch.save(task_embs, file_path)
            benchmark.set_task_embs(task_embs)

            train_datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(train_manip_datasets, task_embs)]
            train_concat_dataset = ConcatDataset(train_datasets)
            self.train_loader = DataLoader(
                train_concat_dataset,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                sampler=RandomSampler(train_concat_dataset),
                persistent_workers=True,
            )

            val_datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(val_manip_datasets, task_embs)]
            val_concat_dataset = ConcatDataset(val_datasets)
            self.val_loader = DataLoader(
                val_concat_dataset,
                batch_size=cfg.eval.batch_size,
                num_workers=cfg.eval.num_workers,
                sampler=RandomSampler(val_concat_dataset),
                persistent_workers=True,
            )

            return shape_meta
    
    def build_model(self, cfg, shape_meta):
        self.model = eval(cfg.policy.policy_type)(cfg, shape_meta)
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)
        self.scheduler = setup_lr_scheduler(self.optimizer, cfg.train.scheduler)

        # resume
        self.start_epoch = 1
        if cfg.train.resume:
            self.start_epoch = self.model.load(cfg.train.resume_path, self.optimizer, self.scheduler)

    def train(self):
        cfg = self.cfg
        self.before_train()

        print('\nTraining...')
        self.fabric.barrier()
        for self.epoch in self.metric_logger.log_every(range(self.start_epoch, cfg.train.n_epochs+1), cfg.eval.eval_every, ""):
            self.before_epoch()
            train_metrics = self.run_epoch()
            self.after_epoch(train_metrics)

        self.after_train()

    def before_train(self):
        cfg = self.cfg
        self.metric_logger = MetricLogger(delimiter=" ")
        self.best_loss_logger = BestAvgLoss(window_size=5)
        None if (cfg.train.dry or not self.fabric.is_global_zero) else init_wandb(cfg)
        # is_main_process = (hasattr(self.fabric, 'is?_global_zero') and self.fabric.is_global_zero) or \
                #   not hasattr(self.fabric, 'is_global_zero')

        # 只有在非 dry run 且是主进程的情况下，才初始化 wandb
        # if not cfg.train.dry and is_main_process:
            # init_wandb(cfg)

        self.losses = MetricMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

    def before_epoch(self):
        pass

    def run_epoch(self):
        tot_loss_dict, tot_items = {}, 0

        self.model.train()
        end = time.time()
        for batch_id, data in enumerate(self.train_loader):
            self.data_time.update(time.time() - end)
            ret_dict = self.forward_backward(data)          # backward propagation

            self.batch_time.update(time.time() - end)
            self.losses.update(ret_dict)
            
            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if (batch_id + 1) % 100 == 0 or len(self.train_loader) < 100:
                    nb_remain = 0
                    nb_remain += len(self.train_loader) - batch_id - 1
                    nb_remain += (self.cfg.train.n_epochs - self.epoch) * len(self.train_loader)
                    eta_seconds = self.batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logging.info("epoch [{0}/{1}][{2}/{3}]\t"
                        "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "eta {eta}\t"
                        "{losses}\t"
                        "lr {lr:.6e}".format(
                            self.epoch,
                            self.cfg.train.n_epochs,
                            batch_id + 1,
                            len(self.train_loader),
                            batch_time=self.batch_time,
                            data_time=self.data_time,
                            eta=eta,
                            losses=self.losses,
                            lr=self.optimizer.param_groups[0]["lr"],
                        ))
                    
            end = time.time()

            if self.cfg.train.debug:
                break
                    
        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        if self.scheduler is not None:
            self.scheduler.step()

        return out_dict
    
    def forward_backward(self, data):
        loss = self.compute_loss(data)

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
        }

        return ret_dict
    
    def compute_loss(self, data, augmentation=None):
        bc_loss = self.model.compute_loss(data, augmentation=augmentation)

        return bc_loss

    def after_epoch(self, train_metrics):
        cfg = self.cfg
        train_metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        self.metric_logger.update(**train_metrics)

        # No evaluation during training

        # if self.fabric.is_global_zero:
        #     None if cfg.train.dry else wandb.log(train_metrics, step=self.epoch)

        #     if self.epoch == 1 or self.epoch % cfg.train.val_freq == 0 or self.epoch == cfg.train.n_epochs:
        #         val_metrics = self.evaluate()
        #         self.metric_logger.update(**val_metrics)
        #         val_metrics = {**val_metrics}
        #         loss_metric = val_metrics["val/loss"]

        #         is_best = self.best_loss_logger.update_best(loss_metric, self.epoch)
        #         if is_best:
        #             # self.model.save(f"{cfg.experiment_dir}/model_best.ckpt", self.epoch, self.optimizer, self.scheduler)
        #             with open(f"{cfg.experiment_dir}/best_epoch.txt", "a") as f:
        #                 f.write(
        #                     "Best epoch: %d, Best %s: %.4f\n"
        #                     % (self.epoch, "loss", self.best_loss_logger.best_loss)
        #                 )
        #         None if cfg.train.dry else wandb.log(val_metrics, step=self.epoch)

        if self.epoch % cfg.train.save_freq == 0:
            self.model.save(f"{cfg.experiment_dir}/model_{self.epoch}.ckpt", self.epoch, self.optimizer, self.scheduler)
            self.rollout_tasks()

        self.fabric.barrier()

    def after_train(self):
        cfg = self.cfg
        # if self.fabric.is_global_zero:
        # self.model.save(f"{cfg.experiment_dir}/model_final.ckpt", self.epoch, self.optimizer, self.scheduler)
        None if cfg.train.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.train.dry else wandb.finish()
        # else:
        self.model.save(f"{cfg.experiment_dir}/model_final.ckpt", self.epoch, self.optimizer, self.scheduler)

    @torch.no_grad()
    def evaluate(self, tag="val"):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0
        self.model.eval()

        print('Evaluating...')
        for data in tqdm(self.val_loader):
            loss = self.compute_loss(data, augmentation=False)

            ret_dict = {
                "loss": loss.item(),
            }

            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if cfg.train.debug:
                break

        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        return out_dict

    def rollout_tasks(self):
        cfg = self.cfg
        if cfg.eval.enable_rollout:
            if cfg.train.debug:
                cfg.env.horizon = 10
                
            results = rollout(cfg, self.rollout_env, self.model, cfg.env.num_env_rollouts // cfg.env.env_num, horizon=cfg.env.horizon)
            
            self.fabric.barrier()
            gathered_results = [{} for _ in range(self.fabric.world_size)]
            dist.all_gather_object(gathered_results, results)
            if self.fabric.is_global_zero:
                gathered_results = merge_results(gathered_results)
                None if cfg.train.dry else wandb.log(gathered_results, step=self.epoch)

                for k in list(gathered_results.keys()):
                    if k.startswith("rollout/vis_"):
                        results.pop(k)

                self.metric_logger.update(**results)

    def reset(self):
        self.policy.reset()

    def inference(self):
        cfg = self.cfg
        for ckp_path in self.ckp_paths_to_eval:
            epoch_name = os.path.basename(ckp_path).split('.ckpt')[0].split('_')[-1]
            video_save_dir = os.path.join(self.eval_result_dir, f"video_{cfg.env.env_name[0]}/epoch_{epoch_name}")
            os.makedirs(video_save_dir, exist_ok=True)

            if self.device == 'cpu':
                results = self.inference_evaluate(
                    checkpoint=ckp_path,
                    video_save_dir=video_save_dir,
                )
                gathered_results = [results]
                gathered_results = merge_results(gathered_results)
                success_metrics = {k: round(v, 3) for k, v in gathered_results.items() if k.startswith("rollout/success_env")}
                save_success_rate(epoch_name, success_metrics, self.summary_file_path)
            else:
                gathered_results = [{} for _ in range(self.fabric.world_size)]
                results = self.inference_evaluate(
                    checkpoint=ckp_path,
                    video_save_dir=video_save_dir,
                )
                self.fabric.barrier()
                dist.all_gather_object(gathered_results, results)

                if self.fabric.is_global_zero:
                    gathered_results = merge_results(gathered_results)
                    success_metrics = {k: round(v, 3) for k, v in gathered_results.items() if k.startswith("rollout/success_env")}
                    save_success_rate(epoch_name, success_metrics, self.summary_file_path)

    def inference_evaluate(self, checkpoint, video_save_dir):
        cfg = self.cfg
        
        # load model
        benchmark = get_benchmark(cfg.data.env_name)(cfg.data.task_order_index)
        shape_meta = get_dataset(dataset_path=os.path.join(cfg.data.root_dir, benchmark.get_task_demonstration(0)),
                                obs_modality=cfg.data.obs.modality, return_shape_meta=True)
        model = eval(cfg.policy.policy_type)(cfg, shape_meta)
        _ = model.load(checkpoint)

        # set optimizer
        cfg.train.optimizer.kwargs.lr = 0.
        optimizer = setup_optimizer(cfg.train.optimizer, model)

        if self.device == 'cpu':
            env_num_each_rank = math.ceil(len(cfg.env.env_name) / 1)
            env_idx_start = env_num_each_rank * 0
            env_idx_end = min(env_num_each_rank * (0 + 1), len(cfg.env.env_name))
        else:
            model, optimizer = self.fabric.setup(model, optimizer)

            # initialize the environments in each rank
            env_num_each_rank = math.ceil(len(cfg.env.env_name) / self.fabric.world_size)
            env_idx_start = env_num_each_rank * self.fabric.global_rank
            env_idx_end = min(env_num_each_rank * (self.fabric.global_rank + 1), len(cfg.env.env_name))

        # save video
        video_writer = VideoWriter(video_save_dir, save_video=True, single_video=False) 

        all_results = []
        for env_idx in range(env_idx_start, env_idx_end):
            print(f"evaluating ckp {checkpoint} on env {cfg.env.env_name[env_idx]} in ({env_idx_start}, {env_idx_end})")
        
            if cfg.eval.debug:
                cfg.env.max_steps = 10
                cfg.env.task_id = [0]
            else:
                cfg.env.task_id = None # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            env = build_env(cfg, img_size=cfg.data.img_size, env_idx_start_end=(env_idx, env_idx+1), **cfg.env)
            result = rollout(
                cfg, env, model, 
                num_env_rollouts=cfg.env.num_env_rollouts // cfg.env.env_num, 
                horizon=cfg.env.max_steps,
                return_wandb_video=False,
                success_vid_first=False, 
                fail_vid_first=False,
                video_writer=video_writer,
                device=self.device,
            )

            if self.device == 'cuda':
                self.fabric.barrier()

            all_results.append(result)
            del env

        all_results = merge_results(all_results, compute_avg=False)

        del model
        del optimizer
        # torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()

        return all_results
