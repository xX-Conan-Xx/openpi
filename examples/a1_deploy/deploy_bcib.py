#!/usr/bin/env python3
"""
A1 Arm Deployment - BCIB (Lightning .ckpt)
- 本地加载 Lightning .ckpt（不依赖 from vla import load_vla）
- 图像预处理参考 CogACTService（resize_image + scale_and_resize）
- 适配你的配置：图像尺寸 128，且模型 obs 需要 states（gripper_states, joint_states）
- 最小闭环：摄像头 -> 模型(get_action/predict_action) -> 动作映射 -> 控制器
"""

import os
import time
import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image


# 兼容 lightning>=2 和旧版 pytorch_lightning
try:
    import lightning as L
    _PLModule = L.LightningModule
except Exception:
    import pytorch_lightning as L
    _PLModule = L.LightningModule


# ======================= 图像处理（与 CogACTService 对齐） =======================

def scale_and_resize(
    image: Image.Image,
    target_size: Tuple[int, int] = (128, 128),
    scale: float = 0.9,
    margin_w_ratio: float = 0.5,
    margin_h_ratio: float = 0.5,
) -> Image.Image:
    """先按比例缩小再偏置裁切，最后缩放回 target_size。"""
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = max(w - new_w, 0)
    margin_h_max = max(h - new_h, 0)
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


def resize_image(
    image: Image.Image,
    size: Tuple[int, int] = (128, 128),
    shift_to_left: int = 0,
) -> Image.Image:
    """宽>高时居中裁成正方形（可向左偏移），再 resize，最后 scale_and_resize。"""
    w, h = image.size
    if h < w:
        left_margin = (w - h) // 2 - shift_to_left
        left_margin = min(max(left_margin, 0), w - h)
        image = image.crop((left_margin, 0, left_margin + h, h))
    else:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))

    image = image.resize(size, resample=Image.LANCZOS)
    image = scale_and_resize(image, target_size=size, scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image


# ======================= Lightning .ckpt 适配器 =======================

class BCIBLightningModel:
    """
    用 Lightning 的 .ckpt 单文件加载模型。
    优先调用 module.get_action(cfg, data)；若不存在，则回落到 module.predict_action(...) 或 forward(...)。
    data 结构对齐你的 obs 配置：
      - 'agentview_rgb': (H,W,3) uint8
      - 'eye_in_hand_rgb': (H,W,3) uint8
      - 'gripper_states': (G,)
      - 'joint_states'  : (J,)
    """
    def __init__(
        self,
        ckpt_path: str,
        module_cls: type,
        module_kwargs: Optional[Dict[str, Any]] = None,
        use_bf16: bool = True,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (use_bf16 and self.device == "cuda") else torch.float32

        self.module: _PLModule = module_cls.load_from_checkpoint(
            ckpt_path, strict=False, map_location=self.device, **(module_kwargs or {})
        )
        self.module.to(self.device).eval()
        try:
            self.module = self.module.to(dtype=self.dtype)
        except Exception:
            pass

    @torch.inference_mode()
    def infer(
        self,
        data: Dict[str, Any],
        instruction: Optional[str],
        unnorm_key: Optional[str],
        do_sample: bool = False,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
    ):
        """
        返回动作（单步 7D 或 多步 Nx7）。
        首选 module.get_action(cfg, data)；回落到 predict_action 或 forward。
        """
        # 首选：get_action(cfg, data)
        if hasattr(self.module, "get_action"):
            cfg_obj = getattr(self.module, "cfg", None)  # 大多实现内部用 self.cfg
            return self.module.get_action(cfg_obj, data)

        # 其次：predict_action(image=..., instruction=..., unnorm_key=...)
        if hasattr(self.module, "predict_action"):
            # 取一张主视角作为 image；如你的实现支持 data dict，请自行放开
            image = data.get("agentview_rgb", None)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return self.module.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=unnorm_key,
                do_sample=do_sample,
                cfg_scale=cfg_scale,
                use_ddim=use_ddim,
                num_ddim_steps=num_ddim_steps,
            )

        # 最后：forward(dict)
        if hasattr(self.module, "forward"):
            return self.module.forward({
                **data,
                "instruction": instruction,
                "unnorm_key": unnorm_key,
                "do_sample": do_sample,
                "cfg_scale": cfg_scale,
                "use_ddim": use_ddim,
                "num_ddim_steps": num_ddim_steps,
            })

        raise AttributeError("Module has no get_action / predict_action / forward; please implement one of them.")


# ======================= A1 部署（仅 bcib 本地 .ckpt） =======================

class A1ArmBCIB:
    def __init__(
        self,
        controller,
        bcib_ckpt: str,
        module_cls: type,
        module_kwargs: Optional[Dict[str, Any]] = None,

        # 推理 / 映射参数
        unnorm_key: Optional[str] = None,
        image_size: Tuple[int, int] = (128, 128),  # ← 配置里的 img_size
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10,
        use_ddim: bool = True,
        use_bf16: bool = True,
        action_dim: int = 7,
        chunk_window: Optional[int] = None,  

        # 姿态控制（ΔRPY）
        use_delta_rpy: bool = False,   
        angle_scale: float = 0.1,      
        angle_step_clip: float = 0.15, 

        # 位置缩放（如果模型输出是 [-1,1] Δ）
        pos_scale: float = 1.0,

        default_prompt: str = "place the green cube and orange into the bowl",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller
        self.pipelines = None

        # 工作空间限幅
        self.pos_limits = {
            "x_min": 0.2, "x_max": 0.4,
            "y_min": -0.02, "y_max": 0.32,
            "z_min": 0.11, "z_max": 0.27,
        }

        # 常用末端位姿预设（xyz + quat + gripper_flag）
        self.position_presets = {
            "pick_bowl": (0.320, 0.12, 0.2456, 0.00, 0.00, 1, 0, 1),
            "env_show":  (0.360, 0.10, 0.35,   0.00, 0.00, 1, 0, 1),
        }

        # 推理配置
        self.image_size = image_size
        self.cfg_scale = cfg_scale
        self.num_ddim_steps = num_ddim_steps
        self.use_ddim = use_ddim
        self.action_dim = action_dim
        self.chunk_window = chunk_window
        self.unnorm_key = unnorm_key
        self.default_prompt = default_prompt
        self.use_delta_rpy = use_delta_rpy
        self.angle_scale = angle_scale
        self.angle_step_clip = angle_step_clip
        self.pos_scale = pos_scale

        # 加载 Lightning .ckpt
        print(f"[BCIB] Loading Lightning ckpt: {bcib_ckpt}")
        print(f"[BCIB] unnorm_key: {self.unnorm_key}")
        self.bcib_model = BCIBLightningModel(
            ckpt_path=bcib_ckpt,
            module_cls=module_cls,
            module_kwargs=module_kwargs,
            use_bf16=use_bf16,
            device=self.device.type,
        )

    # ---------------- Camera ----------------
    def initialize_camera(self):
        self.pipelines = initialize_camera()
        print("Camera pipeline initialized.")
        for _ in range(10):  # warmup
            get_L515_image(self.pipelines)
            get_D435_image(self.pipelines)

    def capture_images(self, step: Optional[int] = None):
        try:
            main_image = get_L515_image(self.pipelines)
            wrist_image = get_D435_image(self.pipelines)
            if main_image is None or wrist_image is None:
                print("Failed to capture image from camera.")
                return None, None, None

            main_image = main_image.convert("RGB")
            wrist_image = wrist_image.convert("RGB")

            save_path = None
            if step is not None:
                save_dir = "/home/luka/Wenkai/visualization/"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"captured_image_{step}.png")
                main_image.save(save_path)

            return main_image, wrist_image, save_path
        except Exception as e:
            print(f"Failed to get image from camera: {e}")
            return None, None, None

    # ------------- Pose utils --------------
    @staticmethod
    def quaternion_to_rpy(quat: List[float]):
        return R.from_quat(quat).as_euler("xyz", degrees=False)

    def preset(self, name: str) -> bool:
        if name not in self.position_presets:
            print(f"Preset {name} not found")
            return False
        self.controller.execute_eef(self.position_presets[name], name)
        return True

    # ------------- 读取机器人状态（gripper & joints） -------------
    def _read_robot_states(self, fallback_joint_dim: int = 7, current_grip: float = 0.0):
        """
        返回 (gripper_states: np.ndarray[G], joint_states: np.ndarray[J])
        会尝试从 controller 读取；若不可用，回退到占位值。
        """
        # gripper
        g = None
        for attr in ("get_gripper_qpos", "get_gripper_state", "get_gripper", "get_gripper_position"):
            if hasattr(self.controller, attr):
                try:
                    g = getattr(self.controller, attr)()
                    break
                except Exception:
                    pass
        if g is None:
            g = np.array([float(current_grip)], dtype=np.float32)
        else:
            g = np.atleast_1d(np.array(g, dtype=np.float32))

        # joints
        j = None
        for attr in ("get_joint_positions", "get_joint_pos", "get_joint_states", "get_joints"):
            if hasattr(self.controller, attr):
                try:
                    j = getattr(self.controller, attr)()
                    break
                except Exception:
                    pass
        if j is None:
            j = np.zeros((fallback_joint_dim,), dtype=np.float32)
        else:
            j = np.atleast_1d(np.array(j, dtype=np.float32))

        return g, j

    # ------------- 构造模型输入（对齐你的 obs 配置） -------------
    def _build_model_input(
        self,
        main_image_pil: Image.Image,
        wrist_image_pil: Image.Image,
        gripper_states: np.ndarray,
        joint_states: np.ndarray,
    ) -> Dict[str, Any]:
        """
        输出 keys:
          - 'agentview_rgb'    : (128,128,3) uint8
          - 'eye_in_hand_rgb'  : (128,128,3) uint8
          - 'gripper_states'   : (G,) float32
          - 'joint_states'     : (J,) float32
        """
        # 按你的配置把相机图像缩放到 128
        main_resized = resize_image(main_image_pil, size=self.image_size)
        wrist_resized = resize_image(wrist_image_pil, size=self.image_size)

        data = {
            "agentview_rgb": np.asarray(main_resized, dtype=np.uint8),
            "eye_in_hand_rgb": np.asarray(wrist_resized, dtype=np.uint8),
            "gripper_states": np.asarray(gripper_states, dtype=np.float32),
            "joint_states": np.asarray(joint_states, dtype=np.float32),
        }
        return data

    # ------------- Inference ---------------
    @torch.inference_mode()
    def _infer_bcib(self, main_img: Image.Image, wrist_img: Image.Image, cur_grip: float) -> np.ndarray:
        # 读取机器人状态（低维）
        g, j = self._read_robot_states(current_grip=cur_grip)

        # 构建 data（图像 128×128 + states）
        data = self._build_model_input(main_img, wrist_img, g, j)

        # 调用模块 infer（优先 get_action(cfg, data)）
        out = self.bcib_model.infer(
            data=data,
            instruction=self.default_prompt,  
            unnorm_key=self.unnorm_key,
            do_sample=False,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )

        # 统一为 ndarray
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        assert arr.shape[-1] == self.action_dim, f"Action dim mismatch: {arr.shape}"
        return arr

    # ------------- Action mapping ----------
    def _apply_pos_limits(self, p):
        p[0] = min(max(p[0], self.pos_limits["x_min"]), self.pos_limits["x_max"])
        p[1] = min(max(p[1], self.pos_limits["y_min"]), self.pos_limits["y_max"])
        p[2] = min(max(p[2], self.pos_limits["z_min"]), self.pos_limits["z_max"])
        return p

    def _process_action(self, action_step, cur_pos, cur_rpy):
        # 位置：Δxyz（可加缩放）
        delta_p = np.asarray(action_step[0:3], dtype=np.float32) * float(self.pos_scale)
        new_pos = (np.asarray(cur_pos, dtype=np.float32) + delta_p).tolist()
        new_pos = self._apply_pos_limits(new_pos)

        # 姿态：默认固定；如需启用 ΔRPY 则叠加（机体系右乘）
        if not self.use_delta_rpy:
            new_rpy = [0.027, -0.01, 3.15]
        else:
            delta_rpy = np.asarray(action_step[3:6], dtype=np.float32) * float(self.angle_scale)
            delta_rpy = np.clip(delta_rpy, -self.angle_step_clip, self.angle_step_clip)
            R_cur = R.from_euler('xyz', cur_rpy)
            R_delta = R.from_euler('xyz', delta_rpy)
            R_new = R_cur * R_delta
            new_rpy = R_new.as_euler('xyz')
            new_rpy = ((new_rpy + np.pi) % (2 * np.pi) - np.pi).tolist()

        # 夹爪：与原约定一致（1 - g）
        g = float(action_step[6])
        gripper = 1.0 - g

        return new_pos, new_rpy, gripper

    # -------------- Main loop --------------
    def run(self, n_iterations=500, loop_interval=0.1, task_name="pick_bowl"):
        if self.pipelines is None:
            self.initialize_camera()

        # 去到预设位姿
        self.preset(task_name)
        preset = self.position_presets[task_name]
        cur_pos = list(preset[0:3])
        cur_quat = preset[3:7]
        cur_rpy = self.quaternion_to_rpy(cur_quat)
        cur_grip = -float(preset[-1])  # 保持你原来的取负号约定

        print("Initial RPY:", cur_rpy)

        step = 0
        try:
            while step < n_iterations:
                t0 = time.time()

                main_image, wrist_image, _ = self.capture_images(step)
                if main_image is None:
                    elapsed = time.time() - t0
                    sleep_time = loop_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    step += 1
                    continue

                # 本地推理（图像 128×128 + states）
                t_inf = time.time()
                all_actions = self._infer_bcib(main_image, wrist_image, cur_grip)
                print(f"Inference time: {time.time() - t_inf:.4f}s")

                # 合并前 K 步，或仅第一步
                if self.chunk_window is not None and self.chunk_window > 1:
                    K = int(min(self.chunk_window, all_actions.shape[0]))
                    merged_prefix = np.sum(all_actions[:K, 0:6], axis=0)
                    grip_cmd = float(all_actions[K - 1, 6])
                    action_step = np.concatenate([merged_prefix, [grip_cmd]], axis=0)
                else:
                    action_step = all_actions[0]

                new_pos, new_rpy, new_grip = self._process_action(action_step, cur_pos, cur_rpy)

                cur_pos, cur_rpy, cur_grip = new_pos, new_rpy, new_grip
                final_action = new_pos + new_rpy + [new_grip]
                print(f"Executing: {final_action}")
                self.controller.execute_eef(final_action)  # 运行时用单参；preset时另有标签

                # 控频（单位：秒）
                elapsed = time.time() - t0
                print(f"Total step time: {elapsed * 1000:.3f} ms")
                sleep_time = loop_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: slower than desired frequency")

                step += 1

            # 收尾：打开夹爪 -> 回预设
            try:
                open_action = cur_pos + cur_rpy + [1.0]
                self.controller.execute_eef(open_action)
                time.sleep(1.0)
            except Exception:
                pass
            self.controller.execute_eef(preset, task_name)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            import traceback
            print(f"Unexpected error: {e}")
            traceback.print_exc()
        finally:
            print("Control loop ended")


# ======================= 入口 =======================

def main():
    from controller_eef import A1ArmController
    from libero_exp.models.bc_diffusion_policy import BCDPPolicy  
    controller = A1ArmController()

    robot = A1ArmBCIB(
        controller=controller,
        bcib_ckpt="/home/luka/xyqjbf/model_final.ckpt", 
        module_cls=BCDPPolicy,                           
        module_kwargs=None,

        unnorm_key=None,                                 
        image_size=(128, 128),                           
        cfg_scale=1.5,
        num_ddim_steps=10,
        use_ddim=True,
        use_bf16=True,
        action_dim=7,
        chunk_window=None,

        use_delta_rpy=False,
        angle_scale=0.1,
        angle_step_clip=0.15,

        pos_scale=1.0,
        default_prompt="place the green cube and orange into the bowl",
    )

    robot.run(
        n_iterations=1000,
        loop_interval=0.1,
        task_name="pick_bowl",
    )


if __name__ == "__main__":
    main()
