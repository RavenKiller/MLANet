import gc
import random
from datetime import datetime

import lmdb
import msgpack_numpy
import numpy as np
import tqdm
import clip
from PIL import Image

import json
from dataclasses import dataclass
import os
import pickle
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import gzip
import json
from pathlib import Path
from nltk.tokenize import word_tokenize

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
from vlnce_baselines.common.utils import Metrics

# from copyreg import pickle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

@dataclass
class EpisodeCls:
    episode_id: int = 0
    episode_len: int = 0
    progress: float = 0.0


class AlkaidEnv:
    def __init__(self, config, mode="val", train_num=1, train_iter=1, extra_step=0):
        self.config = config
        self.ep_idx = -1
        self.step_id = 0
        self.dataset_folder = "/root/autodl-tmp/data/vlntj-ce-extend-{}".format(mode)
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
        with gzip.open("/root/MLANet/data/datasets/R2R_VLNCE_NRSub/train/train.json.gz", "r") as f:
            train_data = json.loads(f.read())
            self.vocab = train_data["instruction_vocab"]

        self.num_envs = 1
        self.old_obs = None
    def _tokenize(self, s, pad_num=200):
        words = word_tokenize(s.lower())
        tokens = []
        for word in words:
            token = int(self.vocab["stoi"].get(word, "1"))
            tokens.append(token)
        if len(tokens)>pad_num:
            tokens = tokens[:pad_num]
        else:
            while len(tokens)<pad_num:
                tokens.append(0)
        return tokens
    def get_current_obs(self, step_id):
        ep_id = self.episode_ids[self.ep_idx]
        rgb = Image.open(
            os.path.join(
                self.dataset_folder, str(ep_id), "rgb", "{}.png".format(step_id)
            )
        ).convert("RGB")
        rgb = np.array(rgb)

        depth = Image.open(
            os.path.join(
                self.dataset_folder, str(ep_id), "depth", "{}.png".format(step_id)
            )
        )
        depth = np.array(depth).astype(np.float32) / 1000.0
        depth = np.clip(depth, 0, 10)
        depth = depth / 10.0
        depth = depth[:, :, np.newaxis]

        with open(
            os.path.join(self.dataset_folder, str(ep_id), "inst", "0.txt"), "r"
        ) as f:
            inst = f.read()
        instruction = {
            "text": inst,
            "tokens": self._tokenize(inst),
            "trajectory_id": ep_id,
        }

        pad_index = 0
        sub_pad_len = 77
        sub_num = 12
        useless_sub = [pad_index] * sub_pad_len
        with open(
            os.path.join(self.dataset_folder, str(ep_id), "sub", "0.txt"), "r"
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
        self.old_obs = {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
            "sub_instruction": sub_instruction,
        }
        return {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
            "sub_instruction": sub_instruction,
        }

    def count_episodes(self):
        return [len(self.episode_ids)]

    def current_episodes(self):
        ep_id = self.episode_ids[self.ep_idx]
        with open(
            os.path.join(self.dataset_folder, str(ep_id), "action", "action.json"), "r"
        ) as f:
            ep_action = json.loads(f.read())
        return [
            EpisodeCls(
                episode_id=self.episode_ids[self.ep_idx],
                episode_len=len(ep_action),
                progress=self.step_id / len(ep_action),
            )
        ]

    def get_prev_action(self):
        ep_id = self.episode_ids[self.ep_idx]
        with open(
            os.path.join(self.dataset_folder, str(ep_id), "action", "action.json"), "r"
        ) as f:
            ep_action = json.loads(f.read())
        return torch.tensor([[ep_action[self.step_id]]], dtype=int)

    def step(self):
        self.step_id += 1
        ep_id = self.episode_ids[self.ep_idx]
        if os.path.exists(
            os.path.join(
                self.dataset_folder, str(ep_id), "rgb", "{}.png".format(self.step_id)
            )
        ):
            obs = self.get_current_obs(self.step_id)
            mask = 0.0
            done = False
            info = {}
        elif self.extra_step>0 and os.path.exists(
            os.path.join(
                self.dataset_folder, str(ep_id), "rgb", "{}.png".format(max(self.step_id-self.extra_step,0))
            )
        ):
            tail_idx = max([int(v.replace(".png","")) for v in os.listdir(os.path.join(self.dataset_folder, str(ep_id), "rgb"))])
            obs = self.get_current_obs(tail_idx)
            mask = 0.0
            done = False
            info = {}
        else:  # next episode
            tail_idx = max([int(v.replace(".png","")) for v in os.listdir(os.path.join(self.dataset_folder, str(ep_id), "rgb"))])
            obs = self.get_current_obs(tail_idx)
            mask = 0.0
            done = True
            info = {}
        return [obs], [mask], [done], [info]

    def reset(self):
        # if self.mode == "train":
        #     np.random.shuffle(self.episode_ids)
        self.ep_idx += 1
        self.step_id = 0
        if self.mode == "train":
            if self.current_iter >= self.train_iter:  # and all
                self.num_envs = 0
                return [self.old_obs]
            else:
                if self.ep_idx >= self.train_num:
                    self.ep_idx = 0
                self.current_iter += 1
            return [self.get_current_obs(self.step_id)]
        else:
            if self.ep_idx >= len(self.episode_ids):
                self.num_envs = 0
                return [self.old_obs]
            return [self.get_current_obs(self.step_id)]

    def finish(self, episode_predictions):
        if self.mode == "train":
            return
        new_folder = (
            self.dataset_folder + str(self.train_num) + "+" + str(self.train_iter)
        )
        os.system(f"rm -r {new_folder}")
        os.system(f"cp -r {self.dataset_folder} {new_folder}")
        for ep_id, ep_data in episode_predictions.items():
            kys = ep_data[0].keys()
            for k in kys:
                data = []
                for v in ep_data:
                    data.append(v[k])
                with open(
                    os.path.join(new_folder, str(ep_id), "action", "action.json"), "r"
                ) as f:
                    gt = json.loads(f.read())
                assert len(data) == len(gt)+self.extra_step
                os.makedirs(os.path.join(new_folder, str(ep_id), k), exist_ok=True)
                with open(
                    os.path.join(new_folder, str(ep_id), k, k + ".json"), "w"
                ) as f:
                    f.write(json.dumps(data))
        n = len(self.episode_ids)
        results = {
            "sr":[0]*n,
            "spl": [0]*n,
            "apa": [0]*n,
            "ndtw": [0]*n,
        }
        from pathlib import Path
        new_folder = Path(new_folder)
        for i in os.listdir(new_folder):
            if i.isnumeric():
                episode_folder = new_folder / i
                with open(episode_folder/"action"/"action.json", "r") as f:
                    gt_action = json.loads(f.read())
                with open(episode_folder/"pred_action"/"pred_action.json", "r") as f:
                    pred_action = json.loads(f.read())
                i = int(i)
                metrics = Metrics(pred_action, gt_action)
                results["sr"][i] = metrics.calc_sr()
                results["spl"][i] = metrics.calc_spl()
                results["apa"][i] = metrics.calc_apa()
                results["ndtw"][i] = metrics.calc_ndtw()
                # metrics.plot_pos(new_folder, i, show=False)
        return results


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

        real_env = AlkaidEnv(config, train_num=config.IL.REAL.train_num, train_iter=config.IL.REAL.train_iter, extra_step=config.IL.REAL.extra_step)

        if not os.path.exists("observation_space.pkl"):
            envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
            observation_space, action_space = self._get_spaces(config, envs=envs)
            import pickle
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

        # TODO: add tornado application and ioloop
    def reset_hiddens(self)
        self.real_rnn_states = torch.zeros(
            real_env.num_envs,
            self.policy.net.num_recurrent_layers,
            self.policy.net.hidden_size,
            device=self.device,
        )
        self.real_prev_actions = torch.zeros(
            real_env.num_envs, 1, device=self.device, dtype=torch.long
        )
        self.real_not_done_masks = torch.zeros(
            real_env.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        real_episode_predictions = defaultdict(list)
        real_observations = real_env.reset()
        real_observations = extract_instruction_tokens(
            real_observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        real_batch = batch_obs(real_observations, self.device)
        real_batch = apply_obs_transforms_batch(real_batch, self.obs_transforms)

        stats_episodes = {}

        num_eps = sum(real_env.count_episodes())
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

    def predict_action(self):

        while len(stats_episodes) < num_eps and real_env.num_envs > 0:
            with torch.no_grad():
                real_actions, real_rnn_states = self.policy.act(
                    real_batch,
                    real_rnn_states,
                    real_prev_actions,
                    real_not_done_masks,
                    deterministic=False,
                )
                # tmp = real_env.get_prev_action().to(self.device)
                real_prev_actions.copy_(real_actions)

            real_observations, _, real_dones, real_infos = real_env.step()
            real_not_done_masks = torch.tensor(
                [[0] if done else [1] for done in real_dones],
                dtype=torch.uint8,
                device=self.device,
            )

            real_current_episodes = real_env.current_episodes()
            for i in range(real_env.num_envs):
                real_episode_predictions[real_current_episodes[i].episode_id].append(
                    {"pred_action": real_actions[i][0].item()}
                )
                if not real_dones[i]:
                    continue
                pbar.update()
                real_observations[i] = real_env.reset()[i]
                # real_rnn_states = torch.zeros(
                #     real_env.num_envs,
                #     self.policy.net.num_recurrent_layers,
                #     self.policy.net.hidden_size,
                #     device=self.device,
                # )
                real_prev_actions = torch.zeros(
                    real_env.num_envs, 1, device=self.device, dtype=torch.long
                )
                # real_not_done_masks = torch.zeros(
                #     real_env.num_envs, 1, dtype=torch.uint8, device=self.device
                # )

            real_observations = extract_instruction_tokens(
                real_observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
            real_batch = batch_obs(real_observations, self.device)
            real_batch = apply_obs_transforms_batch(real_batch, self.obs_transforms)

        envs.close()
        results = real_env.finish(real_episode_predictions)
        mean_results = {k:np.mean(v) for k,v in results.items()}
        print(mean_results)
        path_file = Path(checkpoint_path).name
        fname = os.path.join(
            config.RESULTS_DIR,
            path_file.replace(".pth",".json"),
        )
        with open(fname, "w") as f:
            f.write(json.dumps({"mean":mean_results, "all":results},indent=2))
        if config.use_pbar:
            pbar.close()