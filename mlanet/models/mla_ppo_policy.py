from typing import Optional, Tuple

import abc
from pandas import isna
from simplejson import OrderedDict
from gym import Space, spaces
import numpy as np

import torch
from gym import Space, spaces
from typing import Dict, List, Optional, Union
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN

# from habitat_baselines.utils.common import CategoricalNet, GaussianNet


import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor


from vlnce_baselines.models.encoders import clip_encoders
from vlnce_baselines.common.aux_losses import AuxLosses

# from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from mlanet.models.mla_policy import MLANet


@baseline_registry.register_policy
class MLAPPOPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        model_config,
        **kwargs,
    ):
        super().__init__(
            MLANet(  # type: ignore
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    def extra_load_net(self, config):
        if config.RL.load_net == "legacy":
            ckpt = torch.load(config.RL.net_to_load, map_location="cpu")["state_dict"]
            ckpt_new = self.net.state_dict()
            for k, v in ckpt.items():
                if "net" in k:
                    k = k.replace("net.", "")
                    ckpt_new[k] = v
            self.net.load_state_dict(ckpt_new)
            ckpt_new = OrderedDict()
            for k, v in ckpt.items():
                if "action_distribution" in k:
                    k = k.replace("action_distribution.", "")
                    ckpt_new[k] = v
            self.action_distribution.load_state_dict(ckpt_new)
        elif config.RL.load_net == "ppo":
            ckpt = torch.load(config.RL.net_to_load, map_location="cpu")["state_dict"]
            ckpt_new = self.state_dict()
            for k, v in ckpt.items():
                k = k.replace("actor_critic.", "")
                ckpt_new[k] = v
            self.load_state_dict(ckpt_new)

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        p = cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )
        # p.extra_load_net(config)
        return p

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        return self.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )


class MLANetLegacy(Net):
    def __init__(
        self,
        observation_space: Space,
        model_config: Config,
        num_actions: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()
        self.config = config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)
        if model_config.INSTRUCTION_ENCODER.bidirectional:
            self.low_inst_size = 2 * model_config.INSTRUCTION_ENCODER.hidden_size
        else:
            self.low_inst_size = model_config.INSTRUCTION_ENCODER.hidden_size

        # Init the CLIP encoder
        self.clip_encoder = clip_encoders.CLIPEncoder(
            model_config.CLIP.model_name,
            trainable=model_config.CLIP.trainable,
            downsample_size=model_config.CLIP.downsample_size,
        )
        self.high_inst_size = model_config.CLIP.feature_size

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            clip_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        # Init vision projection layers
        dropout_ratio = model_config.MLA.feature_drop
        self.sub_inst_fc = nn.Sequential(
            # nn.LayerNorm(model_config.CLIP.output_size),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(model_config.CLIP.output_size, model_config.CLIP.feature_size),
            nn.ReLU(inplace=False),
        )
        self.rgb_fc = nn.Sequential(
            # nn.LayerNorm(model_config.CLIP.vit_size),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(model_config.CLIP.vit_size, model_config.CLIP.feature_size),
            nn.ReLU(inplace=False),
        )
        self.depth_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size,
                model_config.DEPTH_ENCODER.feature_size,
            ),
            nn.ReLU(inplace=False),
        )
        self.v_size = (
            model_config.CLIP.feature_size + model_config.DEPTH_ENCODER.feature_size
        )
        self.f_v_size = self.v_size

        # Init action embedding
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the vision RNN state encoder to attend low level instructions
        rnn_input_size = model_config.CLIP.feature_size
        rnn_input_size += model_config.DEPTH_ENCODER.feature_size
        if self.model_config.SEQ2SEQ.encoder_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim
        dropout_ratio_rnn = 0
        self.low_state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_low,
            num_layers=model_config.STATE_ENCODER.num_layers_low,
            dropout=dropout_ratio_rnn,
        )

        # Init the vision RNN state encoder to attend high level instrucitons
        self.high_state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_high,
            num_layers=model_config.STATE_ENCODER.num_layers_high,
            dropout=dropout_ratio_rnn,
        )
        # Vision attend instruction
        self.heads = model_config.MLA.heads
        _hidden_size = model_config.STATE_ENCODER.hidden_size
        self.low_level_attention = nn.MultiheadAttention(
            _hidden_size,
            self.heads,
            batch_first=True,
            kdim=self.low_inst_size,
            vdim=self.low_inst_size,
        )
        self.high_level_attention = nn.MultiheadAttention(
            _hidden_size,
            self.heads,
            batch_first=True,
            kdim=self.high_inst_size,
            vdim=self.high_inst_size,
        )
        self.f_i_size = self.low_inst_size + self.high_inst_size
        self.inst_post = nn.Linear(_hidden_size * 2, self.f_i_size)

        # Instruction attend vision
        self.spatial_attention_rgb = nn.MultiheadAttention(
            self.f_i_size,
            self.heads,
            batch_first=True,
            kdim=model_config.CLIP.vit_size,
            vdim=model_config.CLIP.vit_size,
        )
        self.spatial_attention_depth = nn.MultiheadAttention(
            self.f_i_size,
            self.heads,
            batch_first=True,
            kdim=model_config.DEPTH_ENCODER.single_size,
            vdim=model_config.DEPTH_ENCODER.single_size,
        )
        self.f_v_size = (
            model_config.CLIP.feature_size + model_config.DEPTH_ENCODER.feature_size
        )
        self.visual_post = nn.Linear(self.f_i_size * 2, self.f_v_size)
        # # Init vision-instruction fusion projection layer
        # self.fusion_projection = nn.Sequential(
        #     nn.Linear(self.f_i_size+self.v_size, self.f_v_size),
        #     nn.ReLU()
        # )

        # Init the action decoder RNN
        self.all_feature_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.STATE_ENCODER.hidden_size
            + self.f_i_size
            + self.f_v_size
        )
        if self.model_config.SEQ2SEQ.decoder_prev_action:
            self.all_feature_size += self.prev_action_embedding.embedding_dim

        self.final_input_compress = nn.Sequential(
            nn.Linear(self.all_feature_size, model_config.STATE_ENCODER.hidden_size),
            nn.ReLU(inplace=False),
        )
        self.action_decoder = build_rnn_state_encoder(
            input_size=model_config.STATE_ENCODER.hidden_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        # ??
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((self._hidden_size // 2) ** 0.5))
        )

        # Init the progress monitor
        self.progress_monitor = nn.Linear(model_config.STATE_ENCODER.hidden_size, 1)
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
            nn.init.constant_(self.progress_monitor.bias, 0)

        # Init the peak loss parameter
        # if self.model_config.PEAK_ATTENTION.use:
        self.peak_loss_sigma = model_config.PEAK_ATTENTION.sigma
        self.peak_loss_lambda = model_config.PEAK_ATTENTION.alpha
        self.peak_loss_type = model_config.PEAK_ATTENTION.type
        self.peak_loss_steps = model_config.PEAK_ATTENTION.steps
        self.peak_loss_threshold = model_config.PEAK_ATTENTION.threshold

        self.step_cnt = 0

        # Constants to split rnn states when forward
        self.s1 = self.low_state_encoder.num_recurrent_layers
        self.s2 = (
            self.low_state_encoder.num_recurrent_layers
            + self.high_state_encoder.num_recurrent_layers
        )

        # Pre compute features
        self.rgb_features = None
        self.rgb_seq_features = None
        self.depth_features = None
        self.sub_features = None

        self.feature_spaces = spaces.Dict(
            {
                "f_i": spaces.Box(
                    low=np.finfo(float).min,
                    high=np.finfo(float).max,
                    shape=(self.f_i_size,),
                    dtype=np.float32,
                ),
                "f_v": spaces.Box(
                    low=np.finfo(float).min,
                    high=np.finfo(float).max,
                    shape=(self.f_v_size,),
                    dtype=np.float32,
                ),
            }
        )
        self.train()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return (
            self.low_state_encoder.num_recurrent_layers
            + self.high_state_encoder.num_recurrent_layers
            + self.action_decoder.num_recurrent_layers
        )

    def _peak_loss_schedule(self):
        if self.step_cnt < self.peak_loss_steps:
            self.peak_loss_lambda = (
                self.model_config.PEAK_ATTENTION.alpha
                * (self.step_cnt / self.peak_loss_steps) ** self.peak_loss_type
            )
        else:
            self.peak_loss_lambda = self.model_config.PEAK_ATTENTION.alpha

    def _peak_loss(self, score, mask=None):
        mu = torch.argmax(score, dim=1).unsqueeze(1).repeat((1, score.shape[1]))
        sigma = (
            torch.tensor([self.peak_loss_sigma], dtype=torch.float, device=score.device)
            .unsqueeze(1)
            .repeat(score.shape)
        )
        x = torch.ones_like(score).cumsum(dim=1) - 1
        x = -((x - mu) ** 2) / (2 * sigma**2)
        x = torch.exp(x)
        if mask is not None:
            x[mask] = 0
        e = x / (1e-10 + x.sum(dim=1, keepdim=True))
        loss = F.mse_loss(score, e, reduction="none").sum(dim=1)
        # if torch.rand(1)>self.peak_loss_lambda:
        if torch.rand(1) > self.peak_loss_threshold:
            loss = torch.zeros_like(loss)
        return loss

    def get_rgb_features(self):
        return self.rgb_features

    def get_rgb_seq_features(self):
        return self.rgb_seq_features

    def get_depth_features(self):
        return self.depth_features

    def get_sub_features(self):
        return self.sub_features

    def get_state_features(self):
        return {"f_i": self.f_i, "f_v": self.f_v}

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Embedding
        instruction_embedding = self.instruction_encoder(observations)  # (N,D,L)
        sub_instruction_embedding = self.clip_encoder.encode_text(
            observations
        )  # (N,L,D)
        mask_sub = (sub_instruction_embedding == 0).all(dim=2)
        depth_embedding = self.depth_encoder(observations).flatten(2)  # (N,D,L)
        rgb_embedding, rgb_embedding_seq = self.clip_encoder.encode_image(
            observations
        )  # (N,D), (N,D,L)

        if "rgb_features" not in observations:
            self.rgb_features = rgb_embedding
            self.rgb_seq_features = rgb_embedding_seq
            self.sub_features = sub_instruction_embedding
            self.depth_features = self.depth_encoder.get_depth_features()
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        # Ablation
        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
            # mask_sub = (sub_instruction_embedding==0).all(dim=2)
        if self.model_config.ablate_sub_instruction:
            sub_instruction_embedding = sub_instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0
            rgb_embedding_seq = rgb_embedding_seq * 0

        # Ablation of SSA, the same as ablating sub-instructions
        if self.model_config.ablate_fsa:
            # Absolutely mask sub-instructions
            sub_instruction_embedding = sub_instruction_embedding * 0

        # Pre projection
        sub_instruction_embedding = self.sub_inst_fc(
            sub_instruction_embedding
        )  # (N,L,D)
        instruction_embedding = instruction_embedding.permute(0, 2, 1)  # (N,L,D)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        v_rgb = self.rgb_fc(rgb_embedding)
        v_depth = self.depth_fc(depth_embedding)
        depth_embedding_seq = depth_embedding.permute(0, 2, 1)  # (N,L,D)
        rgb_embedding_seq = rgb_embedding_seq.permute(0, 2, 1)  # (N,L,D)

        # Vision feature compress
        v = torch.cat([v_rgb, v_depth], dim=1)
        if self.model_config.SEQ2SEQ.encoder_prev_action:
            v_in = torch.cat([v, prev_actions], dim=1)
        else:
            v_in = v

        # Low and high level state encoders
        rnn_states_out = rnn_states.detach().clone()
        (
            h_l,
            rnn_states_out[:, 0 : self.s1],
        ) = self.low_state_encoder(
            v_in,
            rnn_states[:, 0 : self.s1],
            masks,
        )
        (
            h_h,
            rnn_states_out[:, self.s1 : self.s2],
        ) = self.high_state_encoder(
            v_in,
            rnn_states[:, self.s1 : self.s2],
            masks,
        )

        # Vision to instruction attention
        # note that convert h_l and h_h [N,D] -> [N,1,D]
        bs = mask_inst.shape[0]
        dim = mask_inst.shape[1]
        mask_inst_ = (
            mask_inst.unsqueeze(1)
            .repeat((1, self.heads, 1))
            .reshape((bs * self.heads, 1, dim))
        )
        dim = mask_sub.shape[1]
        mask_sub_ = (
            mask_sub.unsqueeze(1)
            .repeat((1, self.heads, 1))
            .reshape((bs * self.heads, 1, dim))
        )
        f_i_low, _ = self.low_level_attention(
            h_l.unsqueeze(1),
            instruction_embedding,
            instruction_embedding,
            attn_mask=mask_inst_,
        )
        f_i_high, sub_inst_score = self.high_level_attention(
            h_h.unsqueeze(1),
            sub_instruction_embedding,
            sub_instruction_embedding,
            attn_mask=mask_sub_,
        )
        sub_inst_score = sub_inst_score.squeeze(1)

        # Ablation of MLA
        if self.model_config.ablate_mla:
            f_i_low.fill_(0)
            f_i_low[:, 0, : self.low_inst_size] = torch.mean(
                instruction_embedding, dim=1
            )
            f_i_high.fill_(0)
            f_i_high[:, 0, : self.high_inst_size] = torch.mean(
                sub_instruction_embedding, dim=1
            )

        f_i = torch.cat([f_i_high.squeeze(1), f_i_low.squeeze(1)], dim=1)
        f_i = self.inst_post(f_i)

        # Instruction to vision attention
        f_v_rgb, _ = self.spatial_attention_rgb(
            f_i.unsqueeze(1), rgb_embedding_seq, rgb_embedding_seq
        )
        f_v_depth, _ = self.spatial_attention_depth(
            f_i.unsqueeze(1), depth_embedding_seq, depth_embedding_seq
        )
        f_v = torch.cat([f_v_rgb.squeeze(1), f_v_depth.squeeze(1)], dim=1)
        f_v = self.visual_post(f_v)

        # Construct decoder input
        if self.model_config.SEQ2SEQ.decoder_prev_action:
            all_features = torch.cat([h_h, h_l, prev_actions, f_i, f_v], dim=1)
        else:
            all_features = torch.cat([h_h, h_l, f_i, f_v], dim=1)
        x = self.final_input_compress(all_features)

        # Decoder
        (
            x,
            rnn_states_out[:, self.s2 :],
        ) = self.action_decoder(
            x,
            rnn_states[:, self.s2 :],
            masks,
        )
        # x = torch.tensor([torch.nan, 1.0])

        # AuxLosses
        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.float(),
                observations["progress"].float(),
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss.float(),
                self.model_config.PROGRESS_MONITOR.alpha,
            )
        if self.model_config.PEAK_ATTENTION.use and AuxLosses.is_active():
            self._peak_loss_schedule()
            AuxLosses.register_loss(
                "peak_attention",
                self._peak_loss(sub_inst_score, mask_sub).float(),
                self.peak_loss_lambda,
            )

        self.step_cnt += 1
        return x, rnn_states_out
