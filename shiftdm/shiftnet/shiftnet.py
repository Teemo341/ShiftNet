import torch
import pytorch_lightning as pl
import einops
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from .utils import disabled_train
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class ShiftNetBase(pl.LightningModule):
    def __init__(self, 
                 target_key, 
                 encoder_config, 
                 first_stage_config = None, 
                 ckpt_path=None,
                 ignore_keys=[],
                 learning_rate=1e-5):
        super().__init__()
        self.instantiate_encoder(encoder_config)
        self.instantiate_first_stage(first_stage_config)
        self.encoder_keys = self.encoder.encode_keys if hasattr(self.encoder, 'encode_keys') else [self.encoder.encode_key]
        self.target_key = target_key
        self.learning_rate = learning_rate
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def instantiate_encoder(self, encoder_config):
        """Instantiate the encoder from the given configuration.
        The encoder class are in the ./encoders.py"""
        self.encoder = instantiate_from_config(encoder_config)

    def instantiate_first_stage(self, first_stage_config):
        """Instantiate the first stage model from the given configuration.
        The first stage model is used to encode the input data before passing to the encoder."""
        if first_stage_config is None:
            self.first_stage_model = None
        else:
            self.first_stage_model = instantiate_from_config(first_stage_config)
            self.first_stage_model = self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch, k):
        x = batch[k] # bhwc
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float() # bchw
        return x

    def preprocess_input(self, x_dict: dict):
        """ sd input is [-1,1], but contorl condition is [0,1], so shift condition is also [0,1]
            you may add other operations
        """
        for key in x_dict: # bchw
            x_dict[key] = x_dict[key]*2.0 - 1.0
        return x_dict

    def encode(self, x_dict: dict, first_stage_model: pl.LightningModule = None):
        """Encode the input data using the model's encoder.
            it's optional to use the sd's first stage model for base encoding.
        """
        x_dict = self.preprocess_input(x_dict) # {bchw}
        if first_stage_model is not None:
            with torch.no_grad():
                for key in x_dict:
                    if isinstance(x_dict[key], torch.Tensor):
                        # Ensure the input is a tensor
                        x_dict[key] = first_stage_model.encode(x_dict[key]) # note that vae return a distribution, ,erther use x.sample() or x.mode()
                        if isinstance(x_dict[key], DiagonalGaussianDistribution):
                            x_dict[key] = x_dict[key].mode()
                    else:
                        raise ValueError(f"Input {key} must be a tensor, got {type(x_dict[key])}")

        z = self.encoder(x_dict)
        return z
    
    def decode(self, z: torch.tensor, first_stage_model: pl.LightningModule):
        """Decode the encoded data using the sd's first stage model. Must use to ensure the alignment with sd.
        """
        return first_stage_model.decode(z) # bchw
    
    def loss(self, rec, target):
        """you should define the loss according to the task. The basic situation is the rec_mse. however, in many cases, rec_mse is insufficient. 
        for example, in canny-shiftnet, the loss could be a combination of all-pixel mse and edge mse.
        """
        loss = {}
        loss['mse'] = torch.nn.functional.mse_loss(rec, target) # bchw
        return loss

    def forward(self, batch, return_loss=False):
        x_dict = {key: self.get_input(batch, key) for key in self.encoder_keys} # bchw
        z = self.encode(x_dict, self.first_stage_model)
        rec = self.decode(z, self.first_stage_model)
        if return_loss:
            # combined with loss, define how to train the model
            target = self.get_input(batch, self.target_key)
            loss = self.loss(rec, target)
            return rec, loss
        else:
            return rec
        
    def training_step(self, batch, batch_idx):
        rec, loss = self(batch, return_loss=True)
        for key, value in loss.items():
            self.log(f"train/{key}", value, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        avg_loss = sum(loss.values()) / len(loss)
        self.log("train/loss", avg_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        rec, loss = self(batch, return_loss=True)
        for key, value in loss.items():
            self.log(f"val/{key}", value, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        avg_loss = sum(loss.values()) / len(loss)
        self.log("val/loss", avg_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return avg_loss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = batch.to(self.device)
        x_dict = {key: self.get_input(batch, key) for key in self.encoder_keys} # bchw
        z = self.encode(x_dict, self.first_stage_model)
        rec = self.decode(z, self.first_stage_model)
        target = self.get_input(batch, self.target_key)
        for key in x_dict:
            log[f"{key}_input"] = self.to_rgb(x_dict[key])
        log[f"rec"] = self.to_rgb(rec)
        log[f"target"] = self.to_rgb(target)
        return log

    def to_rgb(self, x):
        channel = x.shape[1]
        if channel <= 3:
            return x
        else:
            if not hasattr(self, f"colorize_{channel}"):
                self.register_buffer(f"colorize_{channel}", torch.randn(3, channel, 1, 1).to(x))
            x = F.conv2d(x, weight=getattr(self, f"colorize_{channel}"))
            x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
            return x