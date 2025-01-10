import clip
import numpy as np
import torch
import torch.nn as nn
from gym import Space, spaces
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import Tensor

from vlnce_baselines.common.utils import single_frame_box_shape


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        downsample_size: int = 3,
        rgb_level: int = -1,
    ) -> None:
        super().__init__()
        self.model, self.preprocessor = clip.load(model_name)
        for param in self.model.parameters():
            param.requires_grad_(trainable)
        self.normalize_visual_inputs = True
        self.normalize_mu = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073])
        self.normalize_sigma = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711])
        self.rgb_embedding_seq = None
        self.ln_rgb = nn.LayerNorm(768)
        self.ln_text = nn.LayerNorm(512)
        if rgb_level == -1:
            self.model.visual.transformer.register_forward_hook(self._vit_hook)
            self.model.transformer.register_forward_hook(self._t_hook)
        else:
            self.model.visual.transformer.resblocks[rgb_level].register_forward_hook(
                self._vit_hook
            )
            self.model.transformer.resblocks[rgb_level].register_forward_hook(
                self._t_hook
            )
        self.text_embedding_seq = None
        self.rgb_downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(downsample_size), nn.Flatten(start_dim=2)
        )

    def _normalize(self, imgs: Tensor) -> Tensor:
        if self.normalize_visual_inputs:
            device = imgs.device
            if self.normalize_sigma.device != imgs.device:
                self.normalize_sigma = self.normalize_sigma.to(device)
                self.normalize_mu = self.normalize_mu.to(device)
            imgs = (imgs / 255.0 - self.normalize_mu) / self.normalize_sigma
            imgs = imgs.permute(0, 3, 1, 2)
            return imgs
        else:
            return imgs

    def _vit_hook(self, m, i, o):
        self.rgb_embedding_seq = o.float()

    def _t_hook(self, m, i, o):
        self.text_embedding_seq = o.float()

    def encode_text(self, observations: Observations) -> Tensor:
        if "sub_features" in observations:
            text_embedding = observations["sub_features"]
        else:
            sub_instruction = observations["sub_instruction"].int()
            bs = sub_instruction.shape[0]
            T = sub_instruction.shape[1]
            ## fast
            # N = sub_instruction.shape[2]
            text_embedding = torch.zeros((bs, T, 512), dtype=torch.float).to(
                sub_instruction.device
            )
            for i in range(bs):
                pad = torch.zeros((1,), dtype=torch.int, device=sub_instruction.device)
                idx = torch.argmin(
                    torch.cat((sub_instruction[i, :, 0], pad))
                )  # effective sub instructions num
                _ = self.model.encode_text(sub_instruction[i][0:idx]).float()
                # LND -> NLD
                text_embedding_seq = self.text_embedding_seq.float().permute(1, 0, 2)
                text_embedding_seq = self.ln_text(text_embedding_seq)
                text_embedding[i][0:idx] = text_embedding_seq[
                    torch.arange(text_embedding_seq.shape[0]),
                    sub_instruction[i][0:idx].argmax(dim=-1),
                ]
        return text_embedding

    def encode_image(
        self, observations: Observations, return_seq: bool = True
    ) -> Tensor:
        if "rgb_features" in observations and "rgb_seq_features" in observations:
            rgb_embedding = observations["rgb_features"]
            rgb_embedding_seq = observations["rgb_seq_features"]
        else:
            rgb_observations = observations["rgb"]
            _ = self.model.encode_image(self._normalize(rgb_observations)).float()
            s = self.rgb_embedding_seq.shape
            # LND -> NLD
            rgb_embedding_seq = self.rgb_embedding_seq.float().permute(1, 0, 2)
            rgb_embedding_seq = self.ln_rgb(rgb_embedding_seq)
            rgb_embedding = rgb_embedding_seq[:, 0, :]
            # NLD -> NDL -> NDHW
            rgb_embedding_seq = (
                rgb_embedding_seq[:, 1:, :].permute(0, 2, 1).reshape(s[1], s[2], 7, 7)
            )
            rgb_embedding_seq = self.rgb_downsample(rgb_embedding_seq)
        if return_seq:
            return (
                rgb_embedding,
                rgb_embedding_seq,
            )  # returns [BATCH x OUTPUT_DIM]
        else:
            return rgb_embedding


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {"depth": single_frame_box_shape(observation_space.spaces["depth"])}
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output
        self.depth_features = None
        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def get_depth_features(self):
        return self.depth_features

    def forward(self, observations: Observations) -> Tensor:
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)
            self.depth_features = x

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
