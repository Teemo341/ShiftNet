from typing import List

import einops
import torch
import numpy as np
import copy
import warnings

from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule
from cldm.cldm import ControlLDM
from shiftdm.ddim_hacked import DDIMSampler

parent_diffusers = {"ldm": LatentDiffusion, 'cldm': ControlLDM}

class ShiftLDM(LatentDiffusion):
    """ ShiftNet can inherit any diffusion model
        some may have different implementations, but it would be easy to adapt"""

    def __init__(self, shift_stage_config, shift_stage_key: List[str] = [], shift_stage_scale: float = 1.0, parent_model = 'ldm', base_locked: bool = True, *args, **kwargs):
        assert parent_model in parent_diffusers, f"parent model must be one of {list(parent_diffusers.keys())}"
        if parent_model != 'ldm':
            self.__class__.__bases__ = (parent_diffusers[parent_model],)  # change the parent class to the specified model
        super().__init__(*args, **kwargs)
        self.instantiate_shift_stage(shift_stage_config)
        self.shift_stage_key = shift_stage_key
        assert len(self.shift_stage_key) > 0, "at least one shift stage key is required"
        self.shift_stage_scale = shift_stage_scale
        self.base_locked = base_locked

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        # shift calibrate scale
        shift_calibrate_scale = 1.0 - 1.0/np.sqrt(alphas)
        self.register_buffer('shift_calibrate_scale', to_torch(shift_calibrate_scale))

    # Instantiate the shift stage model from the config
    def instantiate_shift_stage(self, config):
        self.shift_stage_model = instantiate_from_config(config)
        if hasattr(self.shift_stage_model, 'decoder'):
            self.shift_stage_model.decoder = None  # remove decoder to save memory, only need the encoder
        if hasattr(self.shift_stage_model, 'first_stage_model'):
            self.shift_stage_model.first_stage_model = None  # remove first stage model to save memory, only need the encoder
    def encode_shift_stage(self, x_dict: dict):
        return self.shift_stage_model.encode(x_dict, self.first_stage_model) # enable multi shift stage encoding, return a latent same shape as z
    def get_shift_stage_encoding(self, encoder_posterior):
        return self.get_first_stage_encoding(encoder_posterior) # same as the first stage encoding, apply self.scale_factor * z
 
    # get input, add the shift feature into the condition dict
    def get_input(self, batch, k, return_shift=True, bs=None, *args, **kwargs):
        """enable unnoticeable implementation of get_input while add shift condition"""
        z, c = super().get_input(batch, k, bs=bs, *args, **kwargs)
        if not isinstance(c, dict):
            c = dict(c_crossattn=[c]) # must be dict as controlnet
        if return_shift:
            shift = {key: batch[key] for key in self.shift_stage_key}
            for key in shift:
                if bs is not None:
                    shift[key] = shift[key][:bs]
                shift[key] = shift[key].to(self.device)
                shift[key] = einops.rearrange(shift[key], 'b h w c -> b c h w')
                # shift[key] = shift[key].to(memory_format=torch.contiguous_format).float() #? not sure if this is needed            
            c['shift'] = shift # {key: bchw}
        else:
            c['shift'] = None
        return [z, c]

    def p_mean_variance(self, x, c, t, *args, **kwargs):
        if 'shift' in c and c['shift'] is not None:
            z_shift = self.get_shift_stage_encoding(self.encode_shift_stage(c['shift']))*self.shift_stage_scale # bchw
            c_ = {key: c[key] for key in c if key != 'shift'} # do not pass shift to the base model

            if t == self.num_timesteps - 1: # x_T, add the shift directly to make the expection be shift
                x = x + z_shift

            # x_t is already shifted, but the x_{t-1} should be calibrated
            results = super().p_mean_variance(x, c_, t, *args, **kwargs) # (model_mean, ...)
            shift_calibrate_scale = extract_into_tensor(self.shift_calibrate_scale, t, x.shape) # bchw
            model_mean = results[0] + z_shift * shift_calibrate_scale # calibrated mean to be E(x0)
            results = (model_mean, ) + results[1:] # keep the rest results unchanged
        else:
            if not hasattr(self, '_warned_no_shift'):
                warnings.warn("No shift condition provided, the model will degenerate to the base model.")
                self._warned_no_shift = True
            results = super().p_mean_variance(x, c, t, *args, **kwargs)
        return results
    
    def p_losses(self, x_start, cond, t, noise=None): # adapt from ldm
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if 'shift' in cond and cond['shift'] is not None:
            z_shift = self.get_shift_stage_encoding(self.encode_shift_stage(cond['shift']))*self.shift_stage_scale
            cond_ = {key: cond[key] for key in cond if key != 'shift'}
            x_noisy = x_noisy + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * z_shift # x_start* sqrt_alphas_cumprod, noise*sqrt_one_minus_alphas_cumprod, so shift should be scaled by sqrt_one_minus_alphas_cumprod
        else:
            cond_ = cond
        model_output = self.apply_model(x_noisy, t, cond_)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
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
            log["control"] = c["c_concat"][0] * 2.0 - 1.0 # [-1, 1]
        if exists(c["c_crossattn"]):
            log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        if exists(c["shift"]) and c["shift"] is not None:
            log["shift"] = self.decode_first_stage(self.get_shift_stage_encoding(self.encode_shift_stage(c["shift"]))) # encode and decode to show the shift latent
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
        if self.base_locked:
            self.model.eval()
            self.model.train = disabled_train # disable training for the main model
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            params += list(self.model.parameters())
        if len(params) == 0:
            print("No parameters to optimize")
            return None
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
            self.first_stage_model = self.first_stage_model.cpu() if self.first_stage_model is not None else None
            self.cond_stage_model = self.cond_stage_model.cpu() if self.cond_stage_model is not None else None
            self.shift_stage_model = self.shift_stage_model.cpu() if self.shift_stage_model is not None else None
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu() if self.control_model is not None else None
            self.first_stage_model = self.first_stage_model.cuda() if self.first_stage_model is not None else None
            self.cond_stage_model = self.cond_stage_model.cuda() if self.cond_stage_model is not None else None
            self.shift_stage_model = self.shift_stage_model.cuda() if self.shift_stage_model is not None else None