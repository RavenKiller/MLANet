import gc
import random
from datetime import datetime

import lmdb
import msgpack_numpy
import numpy as np
import tqdm
import clip
import pickle
from PIL import Image

import json
import io
from dataclasses import dataclass
import os
import pickle
from base64 import b64decode
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import gzip
import json
from pathlib import Path
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import request

import jsonlines
import torch
import torch.nn.functional as F
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

# from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
try:
    from habitat_baselines.rl.ddppo.ddp_utils import (
        is_slurm_batch_job,
    )
except ModuleNotFoundError:
    from habitat_baselines.rl.ddppo.algo.ddp_utils import (
        is_slurm_batch_job,
    )
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import Metrics, instruction_cut_clip

# from copyreg import pickle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


app = Flask("alkaid_trainer")


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def load_depth_model():
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    model = torch.hub.load(
        "/share/home/tj90055/.cache/torch/hub/isl-org_ZoeDepth_main",
        "ZoeD_NK",
        source="local",
        pretrained=False,
    )
    model.to("cuda")
    model.eval()
    return model


def center_crop(im):
    width, height = im.size  # Get dimensions
    new_width = min(width, height)
    new_height = new_width

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im


class AlkaidEnv:
    def __init__(self, config, mode="val", train_num=1, train_iter=1, extra_step=0):
        self.config = config
        self.ep_idx = -1
        self.step_id = 0
        self.dataset_folder = "data/vlntj-ce-extend-{}".format(mode)
        self.episode_ids = [
            int(v) for v in os.listdir(self.dataset_folder) if v.isnumeric()
        ]
        self.episode_ids = sorted(self.episode_ids)
        self.mode = mode
        if mode == "train":
            np.random.shuffle(self.episode_ids)
        self.train_num = train_num
        self.train_iter = train_iter
        self.extra_step = extra_step
        self.current_iter = 0
        with gzip.open("data/datasets/R2R_VLNCE_NRSub/train/train.json.gz", "r") as f:
            train_data = json.loads(f.read())
            self.vocab = train_data["instruction_vocab"]

        self.num_envs = 1
        self.depth_model = load_depth_model()

    def _tokenize(self, s, pad_num=200):
        words = word_tokenize(s.lower())
        tokens = []
        for word in words:
            token = int(self.vocab["stoi"].get(word, "1"))
            tokens.append(token)
        if len(tokens) > pad_num:
            tokens = tokens[:pad_num]
        else:
            while len(tokens) < pad_num:
                tokens.append(0)
        return tokens

    def infer_depth(self, image):
        scale = 4.0
        target_size = (256, 256)

        depth_pil = self.depth_model.infer_pil(image, output_type="pil")

        depth_pil = depth_pil.resize(target_size)
        out_arr = np.array(depth_pil)
        out_arr = (
            out_arr * scale
        )  # multiplied by 4, assume it is accurate absolute depth*1000
        # depth_pil = Image.fromarray(out_arr.astype(np.uint16))
        return out_arr

    def get_current_obs(self, ep_id, rgb, depth=None, gt_sub=False):
        tic = time.time()
        rgb_pil = Image.open(io.BytesIO(rgb)).convert("RGB")
        # center crop and resize
        rgb_pil = center_crop(rgb_pil)
        rgb = np.array(rgb_pil.resize((224, 224)))

        time_rgb = time.time() - tic
        tic = time.time()

        if depth:
            depth = Image.open(io.BytesIO(depth))
        else:
            # Due to the resolution, inferred depth may harm the performance
            depth = self.infer_depth(rgb_pil)

        depth = np.array(depth).astype(np.float32) / 1000.0
        depth = np.clip(depth, 0, 10)
        depth = depth / 10.0
        depth = depth[:, :, np.newaxis]
        time_depth = time.time() - tic
        tic = time.time()

        with open(
            os.path.join(self.dataset_folder, str(ep_id), "inst", "0.txt"), "r"
        ) as f:
            inst = f.read()
        instruction = {
            "text": inst,
            "tokens": self._tokenize(inst),
            "trajectory_id": ep_id,
        }
        time_inst = time.time() - tic
        tic = time.time()

        if gt_sub:
            pad_index = 0
            sub_pad_len = 77
            sub_num = 12
            useless_sub = [pad_index] * sub_pad_len
            with open(
                os.path.join(self.dataset_folder, str(ep_id), "sub", "0.txt"),
                "r",
            ) as f:
                sub = f.read()
            sub_tokens = [
                clip.tokenize(v, truncate=True, context_length=77).squeeze(0).tolist()
                for v in sub.split("\n")
            ]
            if len(sub_tokens) > sub_num:
                sub_tokens = sub_tokens[0:sub_num]
            sub_tokens.extend([useless_sub] * (sub_num - len(sub_tokens)))
            sub_instruction = {
                "text": sub,
                "tokens": sub_tokens,
                "trajectory_id": ep_id,
            }
        else:
            sub_data = instruction_cut_clip(
                [{"instruction_text": inst}],
                refine=True,
                append_dot=False,
                keep_subs=True,
            )
            sub = sub_data[0]["sub_instruction"]
            sub_tokens = sub_data[0]["sub_instruction_tokens"]
        sub_instruction = {
            "text": sub,
            "tokens": sub_tokens,
            "trajectory_id": ep_id,
        }
        time_sub = time.time() - tic
        return [
            {
                "rgb": rgb,
                "depth": depth,
                "instruction": instruction,
                "sub_instruction": sub_instruction,
            }
        ], {
            "rgb": time_rgb,
            "depth": time_depth,
            "inst": time_inst,
            "sub": time_sub,
        }


@baseline_registry.register_trainer(name="alkaid_trainer")
class AlkaidTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
            return
        checkpoint_path = self.config.EVAL_CKPT_PATH_DIR

        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt["config"])

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        self.real_env = AlkaidEnv(config)

        if not os.path.exists("observation_space.pkl"):
            envs = construct_envs_auto_reset_false(
                config, get_env_class(config.ENV_NAME)
            )
            observation_space, action_space = self._get_spaces(config, envs=envs)
            with open("observation_space.pkl", "wb") as f:
                pickle.dump(observation_space, f)
            with open("action_space.pkl", "wb") as f:
                pickle.dump(action_space, f)
        else:
            with open("observation_space.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open("action_space.pkl", "rb") as f:
                action_space = pickle.load(f)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()
        self.policy.net.eval()
        self.real_rnn_states = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            self.policy.net.hidden_size,
            device=self.device,
        )
        self.real_not_done_masks = torch.zeros(
            1, 1, dtype=torch.uint8, device=self.device
        )
        self.real_env = AlkaidEnv(config=self.config, mode="val")
        self.step_cnt = 0

        app.add_url_rule(
            "/reset_hiddens",
            "reset_hiddens",
            self.reset_hiddens,
            methods=["POST"],
        )
        app.add_url_rule(
            "/predict_action",
            "predict_action",
            self.predict_action,
            methods=["POST"],
        )
        app.run("0.0.0.0", "32145")

    def reset_hiddens(self):
        # Here, only setting not_done_masks is necessary
        # rnn_states will be zeroed if not_done_masks==0 (in rnn_state_encoder.py)
        # prev_actions will be zeroed if not_done_masks==0 (in mla_policy.py)
        self.real_rnn_states = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            self.policy.net.hidden_size,
            device=self.device,
        )
        self.real_prev_actions = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.real_not_done_masks = torch.zeros(
            1, 1, dtype=torch.uint8, device=self.device
        )
        self.step_cnt = 0
        return {"status": "success"}

    def predict_action(self):
        ep_id = request.form.get("ep_id")
        rgb = request.form.get("rgb")
        depth = request.form.get("depth", None)
        path_len = len(
            list(
                os.listdir(
                    os.path.join(self.real_env.dataset_folder, str(ep_id), "rgb")
                )
            )
        )
        tic = time.time()
        rgb = b64decode(rgb)
        if depth:
            depth = b64decode(depth)
        time_info_decode = time.time() - tic
        tic = time.time()

        real_observations, time_info_detail = self.real_env.get_current_obs(
            ep_id, rgb, depth
        )
        time_info_observation = time.time() - tic
        tic = time.time()
        real_observations = extract_instruction_tokens(
            real_observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        real_batch = batch_obs(real_observations, self.device)
        real_batch = apply_obs_transforms_batch(real_batch, self.obs_transforms)
        with torch.no_grad():
            real_actions, self.real_rnn_states = self.policy.act(
                real_batch,
                self.real_rnn_states,
                self.real_prev_actions,
                self.real_not_done_masks,
                deterministic=False,  # May affect multi-round experiments
            )
            # tmp = real_env.get_prev_action().to(self.device)
            self.real_prev_actions.copy_(real_actions)
        self.real_not_done_masks = torch.ones(
            1, 1, dtype=torch.uint8, device=self.device
        )
        time_info_forward = time.time() - tic
        action_out = real_actions.cpu().item()
        if self.step_cnt >= path_len + 3 or self.step_cnt > 100:
            action_out = 0
        self.step_cnt += 1
        return {
            "status": "success",
            "action": action_out,
            "time_info": {
                "decode_b64": time_info_decode,
                "observation": time_info_observation,
                "observation_detail": time_info_detail,
                "forward": time_info_forward,
            },
        }
