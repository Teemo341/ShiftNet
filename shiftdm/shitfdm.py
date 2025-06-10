from typing import Union

import einops
import torch
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.modules.diffusionmodules.util import conv_nd, linear, zero_module, timestep_embedding
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from shiftdm.ddim_hacked import DDIMSampler


class diffusion_example(LatentDiffusion):
    """A simple example of a Latent Diffusion model
    ShiftNet can inherit any diffusion model with the following methods
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ShiftLDM(LatentDiffusion):
    """ ShiftNet can inherit any diffusion model """

    def __init__(self, shift_stage_config, shift_stage_key:list[str]=[], shift_stage_scale:float = 1.0, sd_locked:bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instantiate_shift_stage(shift_stage_config)
        self.shift_stage_key = shift_stage_key
        assert len(self.shift_stage_key) > 0, "at least one shift stage key is required"
        self.shift_stage_scale = shift_stage_scale
        self.sd_locked = sd_locked

    # Instantiate the shift stage model from the config
    def instantiate_shift_stage(self, config):
        self.shift_stage_model = instantiate_from_config(config)
    def encode_shift_stage(self, x_dict: dict):
        return self.shift_stage_model.encode(x_dict) # enable multi shift stage encoding, return a latent same shape as z
    def get_shift_stage_encoding(self, encoder_posterior):
        return self.get_first_stage_encoding(encoder_posterior) # same as the first stage encoding
    
    def get_mu_scale(self, t):
        """ At time t, mu_z_noised/ mu_z is um_scale. Shiftnet add the (1-mu_scale) to achieve constant scale.
        """
        mu_scale = self.sqrt_alphas_cumprod[t] # In ddpm, mu_scale is sqrt_alphas_cumprod

        return mu_scale

    
    # get input, add the shift feature into the condition dict
    def get_input(self, batch, k, return_shift=True, bs=None, *args, **kwargs):
        """enable unnoticeable implementation of get_input while add shift condition"""
        z, c = super().get_input(batch, k, bs=bs, *args, **kwargs)
        if return_shift:
            assert isinstance(c, dict), "condition must be a dict"
            shift = {key: batch[key] for key in self.shift_stage_key}
            for key in shift:
                if bs is not None:
                    shift[key] = shift[key][:bs]
                shift[key] = shift[key].to(self.device)
                shift[key] = einops.rearrange(shift[key], 'b h w c -> b c h w')
                # shift[i] = shift[i].to(memory_format=torch.contiguous_format).float() #? not sure if this is needed
            
            # encode the shift image #! should check the grad
            z_shift = self.get_shift_stage_encoding(self.encode_shift_stage(shift)) # same shape as z
            c['shift'] = z_shift
        else:
            c['shift'] = None
        return [z, c]
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """ pop the shift feature, scale, add to the input, and remove from the output"""
        assert isinstance(cond, dict)
        z_shift = None

        # add shift to the input
        if 'shift' in cond and cond['shift'] is not None:
            z_shift = cond.pop('shift')
            mu_scale = self.get_mu_scale(t)
            z_shift = z_shift * (1.0-mu_scale) * self.shift_stage_scale
            x_noisy = x_noisy + z_shift # add shift to make mu constant

        if self.sd_locked: #TODO no sure if the no_grad is necessary and possible
            with torch.no_grad():
                model_output = super().apply_model(x_noisy, t, cond, *args, **kwargs) # apply the model
        else:
            model_output = super().apply_model(x_noisy, t, cond, *args, **kwargs)

        # remove the shift from output so that the output still follows the original model output
        if z_shift is not None:
            if isinstance(model_output, tuple):
                model_output = (model_output[0] - z_shift, *model_output[1:])
            else:
                model_output = model_output - z_shift

        return model_output
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=True, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        if exists(c["c_concat"]):
            log["control"] = c["c_concat"][0] * 2.0 - 1.0
        if exists(c["c_crossattn"]):
            log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        if exists(c["shift"]) and c["shift"] is not None:
            log["shift"] = self.decode_first_stage(c["shift"])
        log["reconstruction"] = self.decode_first_stage(z)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if exists(c["shift"]) and c["shift"] is not None:
                x_samples = self.decode_first_stage(samples+c['shift']*self.shift_stage_scale) # check if the shift is added to the sample
                log["samples_shift"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
                if exists(c["shift"]) and c["shift"] is not None:
                    z_denoise_row = [z + c["shift"] * self.shift_stage_scale for z in z_denoise_row]  # add shift to the denoise row
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    log["denoise_row_shift"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c["c_concat"][0] if exists(c["c_concat"]) else None
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "shift": c["shift"] if exists(c["shift"]) else None}
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.shift_stage_model.parameters())
        if self.sd_locked:
            self.model.eval()
            self.model.train = disabled_train # disable training for the main model
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            params += list(self.model.parameters())
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

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda() if self.control_model is not None else None
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
            self.shift_stage_model = self.shift_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu() if self.control_model is not None else None
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
            self.shift_stage_model = self.shift_stage_model.cuda()