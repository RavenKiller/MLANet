from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

# import numpy as np
import torch
from gym import spaces
import numbers
import numpy as np
import copy
import json
import random
import re
import time

import clip
import nltk
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
from dtw import dtw

# from habitat.core.utils import try_cv2_import
from habitat.utils import profiling_wrapper

# from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import DictTree, TensorDict

# from gym.spaces import Box
# from habitat import logger
# from habitat.core.dataset import Episode


# from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# from PIL import Image
# from torch import Size, Tensor
# from torch import nn as nn

PAD_LEN = 25
PAD_SEQ = [0] * PAD_LEN
MAX_SUB = 10


def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    sub_instruction_sensor_uuid: str = None,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if sub_instruction_sensor_uuid is None:
        sub_instruction_sensor_uuid = "sub_" + instruction_sensor_uuid
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = np.array(
                observations[i][instruction_sensor_uuid][tokens_uuid]
            )
            observations[i][sub_instruction_sensor_uuid] = np.array(
                observations[i][sub_instruction_sensor_uuid][tokens_uuid]
            )
        # else:
        #     break
    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )


# My cumstomized batch_obs() function
@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


# @torch.no_grad()
# @profiling_wrapper.RangeContext("batch_obs_new")
# def batch_obs_new(
#     observations: List[DictTree],
#     device: Optional[torch.device] = None,
#     cache: Optional[ObservationBatchingCache] = None,
# ) -> TensorDict:
#     r"""Transpose a batch of observation dicts to a dict of batched
#     observations.

#     Args:
#         observations:  list of dicts of observations.
#         device: The torch.device to put the resulting tensors on.
#             Will not move the tensors if None
#         cache: An ObservationBatchingCache.  This enables faster
#             stacking of observations and cpu-gpu transfer as it
#             maintains a correctly sized tensor for the batched
#             observations that is pinned to cuda memory.

#     Returns:
#         transposed dict of torch.Tensor of observations.
#     """
#     batch_t: TensorDict = TensorDict()
#     if cache is None:
#         batch: DefaultDict[str, List] = defaultdict(list)

#     obs = observations[0]
#     # Order sensors by size, stack and move the largest first
#     sensor_names = sorted(
#         obs.keys(),
#         key=lambda name: 1
#         if isinstance(obs[name], numbers.Number)
#         else np.prod(obs[name].shape),
#         reverse=True,
#     )

#     for sensor_name in sensor_names:
#         for i, obs in enumerate(observations):
#             sensor = obs[sensor_name]
#             if cache is None:
#                 batch[sensor_name].append(torch.as_tensor(sensor))
#             else:
#                 if sensor_name not in batch_t:
#                     batch_t[sensor_name] = cache.get(
#                         len(observations),
#                         sensor_name,
#                         torch.as_tensor(sensor),
#                         device,
#                     )

#                 # Use isinstance(sensor, np.ndarray) here instead of
#                 # np.asarray as this is quickier for the more common
#                 # path of sensor being an np.ndarray
#                 # np.asarray is ~3x slower than checking
#                 if isinstance(sensor, np.ndarray):
#                     batch_t[sensor_name][i] = sensor
#                 elif torch.is_tensor(sensor):
#                     batch_t[sensor_name][i].copy_(sensor, non_blocking=True)
#                 # If the sensor wasn't a tensor, then it's some CPU side data
#                 # so use a numpy array
#                 else:
#                     batch_t[sensor_name][i] = np.asarray(sensor)

#         # With the batching cache, we use pinned mem
#         # so we can start the move to the GPU async
#         # and continue stacking other things with it
#         if cache is not None:
#             # If we were using a numpy array to do indexing and copying,
#             # convert back to torch tensor
#             # We know that batch_t[sensor_name] is either an np.ndarray
#             # or a torch.Tensor, so this is faster than torch.as_tensor
#             if isinstance(batch_t[sensor_name], np.ndarray):
#                 batch_t[sensor_name] = torch.from_numpy(batch_t[sensor_name])

#             batch_t[sensor_name] = batch_t[sensor_name].to(
#                 device, non_blocking=True
#             )

#     if cache is None:
#         for sensor in batch:
#             batch_t[sensor] = torch.stack(batch[sensor], dim=0)

#         batch_t.map_in_place(lambda v: v.to(device))

#     return batch_t

EPISODE_NUM = 13
FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


def euclidean_distance(pos_a, pos_b) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


class Metrics:
    def __init__(self, pred_action, gt_action):
        self.sr_threshold = 1.5
        self.turn_angle = 15

        self.pred_action = pred_action
        self.gt_action = gt_action
        self.pred_position, self.pred_heading = self.calc_pos(pred_action)
        self.gt_position, self.gt_heading = self.calc_pos(gt_action)

    def calc_pos(self, action):
        current_position = (0.0, 0.0)
        current_heading = 0
        res_position = [current_position]
        res_heading = [current_heading]
        for a in action:
            if a == 1:
                dx = 0.25 * np.cos(np.deg2rad(current_heading))
                dy = 0.25 * np.sin(np.deg2rad(current_heading))
                current_position = (
                    current_position[0] + dx,
                    current_position[1] + dy,
                )
                # print(np.sqrt(dx*dx+dy*dy))
            elif a == 2:
                dw = self.turn_angle
                current_heading = current_heading + dw
                if current_heading >= 360:
                    current_heading = current_heading - 360
            elif a == 3:
                dw = -self.turn_angle
                current_heading = current_heading + dw
                if current_heading < 0:
                    current_heading = current_heading + 360
            res_position.append(current_position)
            res_heading.append(current_heading)
        return res_position, res_heading

    def calc_len(self, position):
        coors = np.array(position)
        diff = coors[:-1] - coors[1:]
        dist = np.sum(np.linalg.norm(diff, axis=1, ord=2))
        return dist

    def calc_sr(self):
        pred_coor = self.pred_position[-1]
        gt_coor = self.gt_position[-1]
        diff = np.array(pred_coor) - np.array(gt_coor)
        dist = np.linalg.norm(diff, ord=2)
        if dist <= self.sr_threshold:
            return 1.0
        else:
            return 0.0

    def calc_spl(self):
        sr = self.calc_sr()
        pred_l = self.calc_len(self.pred_position)
        gt_l = self.calc_len(self.gt_position)
        spl = sr * gt_l / max(pred_l, gt_l)
        return spl

    def calc_apa(self):
        pred_action = np.array(self.pred_action, dtype=int)
        gt_action = np.array(self.gt_action, dtype=int)
        if len(pred_action) < len(gt_action):
            pred_action = np.pad(
                pred_action,
                pad_width=((0, len(gt_action) - len(pred_action))),
                mode="constant",
            )
        else:
            pred_action = pred_action[: len(gt_action)]
        return float(np.sum(pred_action == gt_action)) / len(gt_action)

    def calc_ndtw(self):
        dtw_distance = dtw(
            self.pred_position, self.gt_position, dist=euclidean_distance
        )[0]

        nDTW = np.exp(-dtw_distance / (len(self.pred_position) * self.sr_threshold))
        return nDTW

    def plot_pos(self, data_folder, idx=-1, show=False):
        plt.figure(dpi=600, figsize=(5.4, 4.8))
        pred_position = np.array(self.pred_position)
        gt_position = np.array(self.gt_position)
        min_v = np.min(gt_position)
        pred_position = pred_position - min_v
        gt_position = gt_position - min_v
        max_v = np.max(gt_position)
        pred_position = pred_position / max_v
        gt_position = gt_position / max_v
        x_min = min(gt_position[:, 0].min(), pred_position[:, 0].min())
        y_min = min(gt_position[:, 1].min(), pred_position[:, 1].min())
        x_max = max(gt_position[:, 0].max(), pred_position[:, 0].max())
        y_max = max(gt_position[:, 1].max(), pred_position[:, 1].max())
        d_max = max(x_max - x_min, y_max - y_min)
        plt.plot(pred_position[1:, 0], pred_position[1:, 1], marker="o", linewidth=2)
        plt.plot(gt_position[:, 0], gt_position[:, 1], linewidth=2)
        time.sleep(1)
        ms = 144
        plt.scatter(gt_position[0, 0], gt_position[0, 1], marker="*", color="k", s=ms)
        plt.scatter(gt_position[-1, 0], gt_position[-1, 1], marker="^", color="k", s=ms)
        plt.legend(["MLANet", "Ground truth", "Start", "Target"], fontsize=18)
        # plt.axis('scaled')
        plt.xlim([x_min - 0.04, x_min + d_max + 0.04])
        plt.ylim([y_min - 0.04, y_min + d_max + 0.04])
        plt.axis(False)
        if idx > -1:
            plt.savefig(data_folder / "{}/vis.png".format(idx), dpi=600)
        if show:
            plt.show()
        plt.close()


def instruction_cut_clip(
    episodes,
    append_dot=False,
    keep_subs=True,
    refine=False,
    split_func=None,
    lower_all=False,
):
    """
    Params:
        episodes: a dataset list, [{"instruction_text":""}]
        append_dot: whether to add a "." at the end of sub-instructions
        keep_subs: whether to keep sub-instruction text after processing
        refine: whether to use proposed refine processing
        split_func: the function used to cut instructions, default is `nltk.sent_tokenize`
        lower_all: whether to lower all characters in sub-instructions.
    Return:
        train_data with "sub_instruction_tokens" and "sub_instruction"
    """
    if split_func is None:
        split_func = nltk.sent_tokenize
    train_data = copy.deepcopy(episodes)
    # pre process
    char_pattern = re.compile(r"[a-zA-Z]")
    for i, item in enumerate(train_data):
        inst = item["instruction_text"]
        inst = inst.strip()
        start_idx = 0
        while not char_pattern.search(inst[start_idx]):
            start_idx += 1
        inst = inst[start_idx:]
        if lower_all:
            inst = inst.lower()
        train_data[i]["instruction_text"] = (
            inst.replace("...", ".")
            .replace("..", ".")
            .replace(".", ". ")
            .replace("  ", " ")
        )

    # cut by nltk
    pattern = re.compile(r"\r\n")
    for i, item in enumerate(train_data):
        inst = item["instruction_text"]
        res = []
        now = pattern.split(inst)
        for v in now:
            res.extend(split_func(v))
        train_data[i]["sub_instruction"] = [
            piece.strip() for piece in res if piece.strip()
        ]
    # refine
    if refine:
        punctuation_list = [",", "."]
        char_pattern = re.compile(r"[a-zA-Z]+")

        def judge_verb(word):
            const_verbs = ["wait", "turn", "walk", "stop"]
            if "VB" in word[1]:
                return True
            if word[0] in const_verbs:
                return True
            return False

        for i, item in enumerate(train_data):
            new_sub = []
            for k, piece in enumerate(item["sub_instruction"]):
                word_list = nltk.pos_tag(nltk.word_tokenize(piece))
                tmp = ""
                for x, word in enumerate(word_list):
                    if (
                        word[0].lower() == "and"
                        or word[0] == ","
                        or word[0].lower() == "then"
                    ) and (x + 1 < len(word_list) and judge_verb(word_list[x + 1])):
                        if tmp and char_pattern.search(tmp):
                            new_sub.append(tmp)
                        if word[0].lower() == "and" or word[0].lower() == "then":
                            tmp = word[0]
                        else:
                            tmp = ""

                    elif (word[0] == "and" or word[0] == ",") and (
                        x + 1 < len(word_list) and word_list[x + 1][0] == "then"
                    ):
                        if tmp:
                            new_sub.append(tmp)
                        if word[0].lower() == "and" or word[0].lower() == "then":
                            tmp = word[0]
                        else:
                            tmp = ""
                    else:
                        if not tmp or word[0] in punctuation_list:
                            tmp += word[0]
                        else:
                            tmp += " " + word[0]
                if tmp:
                    new_sub.append(tmp)
            train_data[i]["sub_instruction"] = new_sub

    # post process and generate tokens
    char_pattern = re.compile(r"[a-zA-Z]")
    pad_index = 0
    sub_pad_len = 77  # 0.05%
    sub_num = 12  # 0.04%
    useless_sub = [pad_index] * sub_pad_len
    for i, item in enumerate(train_data):
        tokens_all = []
        tokens_split = []
        for k, piece in enumerate(item["sub_instruction"]):
            piece = piece.strip()
            assert piece
            idx = len(piece) - 1
            while idx >= 0 and piece[idx] in [".", ","]:
                idx -= 1
            if append_dot:
                piece = piece[0 : (idx + 1)] + "."
            else:
                piece = piece[0 : (idx + 1)]
            piece = piece.replace("``", '"').replace("''", '"')
            train_data[i]["sub_instruction"][k] = piece
            piece_tokens = clip.tokenize(piece, truncate=True).squeeze(0).tolist()
            tokens_split.append(piece_tokens)
        if len(tokens_split) > sub_num:
            tokens_split = tokens_split[0:sub_num]
        tokens_split.extend([useless_sub] * (sub_num - len(tokens_split)))

        train_data[i]["sub_instruction_tokens"] = tokens_split
        if not keep_subs:
            del item["sub_instruction"]
    return train_data
