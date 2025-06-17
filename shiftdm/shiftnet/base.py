import torch
import pytorch_lightning as pl
import einops

from ldm.util import instantiate_from_config
from .utils import disabled_train
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


DATA_EXAMPLE = {'shift1': torch.randn(1, 3, 256, 256),} # all bchw

class ShiftNetBase(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-5, sd_locked=True, only_mid_control=False):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control

    def instantiate_encoder(self, encoder_config):
        """Instantiate the encoder from the given configuration.
        The encoder class are in the ./encoders.py"""
        self.encoder = instantiate_from_config(encoder_config)

    @torch.no_grad()
    def preprocess_input(self, x_dict: dict):
        """ sd input is [-1,1], but contorl condition is [0,1], so shift condition is also [0,1]
            you may add other operations
        """
        for key in x_dict:
            x_dict[key] = x_dict[key]*2.0 - 1.0
        return x_dict

    def encode(self, x_dict: dict, first_stage_model: pl.LightningModule = None):
        """Encode the input data using the model's encoder.
            it's optional to use the sd's first stage model for base encoding.
        """
        x_dict = self.preprocess_input(x_dict)
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

        x_dict = self.encoder(x_dict)
        return x_dict
    
    def decode(self, z: torch.tensor, first_stage_model: pl.LightningModule):
        """Decode the encoded data using the sd's first stage model. Must use to ensure the alignment with sd.
        """
        return first_stage_model.decode(z)

    def configure_optimizers(self):
        return pl.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass