import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class diffusion_example(LatentDiffusion):
    """A simple example of a Latent Diffusion model
    ShiftNet can inherit any diffusion model with the following methods
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ShiftLDM(LatentDiffusion):
    """ ShiftNet can inherit any diffusion model """

    def __init__(self, shift_stage_config, shift_key, shift_scale, global_average_pooling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling