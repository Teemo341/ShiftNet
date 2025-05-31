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
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
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

    def __init__(self, shift_stage_config, shift_stage_key, shift_stage_scale = 1.0, sd_locked = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instantiate_shift_stage(shift_stage_config)
        self.shift_stage_key = shift_stage_key
        self.shift_stage_scale = shift_stage_scale
        self.sd_locked = sd_locked

    def instantiate_shift_stage(self, config):
        self.shift_stage_model = instantiate_from_config(config)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def encode_shift_stage(self, x_list: list):
        """
        Encodes the input using the shift stage model.
        Args:
            x_list (list): List of inputs to encode.
        Returns:
            torch.Tensor: Encoded output from the shift stage model.
        """
        return self.shift_stage_model.encode(x_list)

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()