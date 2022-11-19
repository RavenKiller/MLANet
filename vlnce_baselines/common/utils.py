from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

# import numpy as np
import torch
from gym import spaces
import numbers
import numpy as np


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
            observations[i][instruction_sensor_uuid] = np.array(observations[i][
                instruction_sensor_uuid
            ][tokens_uuid])
            observations[i][sub_instruction_sensor_uuid] = np.array(observations[i][
                sub_instruction_sensor_uuid
            ][tokens_uuid])
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
