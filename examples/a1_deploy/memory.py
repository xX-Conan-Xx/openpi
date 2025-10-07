import numpy as np

'''
author: Wenkai 
Class for trajectory matching, runhao's model
'''

class TrajMatcher:
    """最小化重用逻辑：给定当前轨迹，找到参考库里代价最低的轨迹片段。"""

    def __init__(self, states_ref, segments_ref, actions_ref, window: int = 20):
        self.states_ref   = [np.asarray(s) for s in states_ref]
        self.segments_ref = segments_ref
        self.actions_ref  = actions_ref
        self.window       = window
        # 记忆上一次匹配结果，用于“卡住”检测
        self.prev_traj, self.prev_ref, self.stuck = 0, 0, 0

    # --------------------------------------------------------------------- #
    def match(self, cur_states: list, cur_seg: int) -> dict:
        cur = np.asarray(cur_states)
        best = {"traj": -1, "ref": -1, "seg": cur_seg, "cost": np.inf}

        for i, ref in enumerate(self.states_ref):
            for ref_idx in self._search_indices(ref.shape[0]):
                if abs(self.segments_ref[i][ref_idx] - cur_seg) > 1:
                    continue
                ref_traj, cur_traj = self._cut_trajs(ref, cur, ref_idx)
                L = min(len(cur_traj), len(ref_traj))
                cost = np.linalg.norm(cur_traj[:L] - ref_traj[:L])                
            if cost < best["cost"]:
                    best.update({"traj": i, "ref": ref_idx,
                                 "seg": self.segments_ref[i][ref_idx],
                                 "cost": cost})
        self._update_stuck(best)
        return best

    # ------------------------- internal helpers -------------------------- #
    def _search_indices(self, ref_len):
        w = self.window
        return range(5, w + 10) if ref_len <= w else range(w + 1, ref_len)

    def _cut_trajs(self, ref, cur, ref_idx):
        w = self.window
        if cur.shape[0] <= w:
            return ref[:ref_idx, :3], cur[:, :3]
        return ref[ref_idx - w:ref_idx, :3], cur[-w:, :3]

    def _update_stuck(self, best):
        if (best["traj"] == self.prev_traj and
                abs(best["ref"] - self.prev_ref) <= 1):
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_traj, self.prev_ref = best["traj"], best["ref"]


# ===================================================================== #
def infer_action(t: int,
                 main_img, wrist_img,
                 current_state,                 # 当前关节+EEF状态
                 state_hist: list,              # <=== 新增：历史轨迹列表
                 default_prompt: str,
                 client,
                 matcher: TrajMatcher,
                 task_index: int,
                 current_segment: int) -> np.ndarray:
    """
    state_hist: 迄今为止所有 step 的 `current_state` 列表，用于匹配参考轨迹。
    """

    img_array = np.asarray(main_img).astype(np.uint8)[None, ...]
    img_wrist_array = np.asarray(wrist_img).astype(np.uint8)[None, ...]
    if t < 5:
        return _call_model(client, img_array, img_wrist_array,
                           current_state, (19, 99, 0), default_prompt)

    # ---------- 2. 匹配参考轨迹 ------------------------------------ #
    best = matcher.match(cur_states=state_hist, cur_seg=current_segment)
    current_segment = best["seg"]            # 更新 segment

    need_replan = best["cost"] > 0.02 or matcher.stuck > 1
    if need_replan:
        return _call_model(client, img_array, img_wrist_array,
                           current_state, (19, 99, 0), default_prompt)

    # ---------- 3. 双推理 + 加权融合 ------------------------------- #
    act_prompt = _call_model(client, img_array, img_wrist_array,
                             current_state, (best["seg"], best["traj"], task_index),
                             default_prompt)
    act_plain  = _call_model(client, img_array, img_wrist_array,
                             current_state, (19, 99, 0), default_prompt)

    ref_chunk  = np.asarray(matcher.actions_ref[best["traj"]]
                            [best["ref"]: best["ref"] + 50])
    sim_prompt = -np.linalg.norm(ref_chunk - act_prompt[:len(ref_chunk)])
    sim_plain  = -np.linalg.norm(ref_chunk - act_plain [:len(ref_chunk)])

    weights = np.exp([sim_prompt, sim_plain])
    alpha   = weights[0] / weights.sum()
    return alpha * act_prompt + (1 - alpha) * act_plain



# ===================================================================== #
def _call_model(client, img_arr, img_wrist_arr, state, extra_tags, prompt):
    elem = {
        "observation/image":        img_arr[0],
        "observation/wrist_image":  img_wrist_arr[0],
        "observation/state":        np.concatenate((state, extra_tags), dtype=np.float32),
        "prompt":                   prompt,
    }
    return client.infer(elem)
    