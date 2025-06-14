import pytorch_lightning as pl

from ldm.util import instantiate_from_config
from .utils import disabled_train


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

    def instantiate_decoder(self, decoder_config):
        """The decoder must be the same as the first stage decoder."""
        model = instantiate_from_config(decoder_config)
        self.decoder = model.decoder
        self.decoder.train = disabled_train
        for param in self.decoder.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return pl.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass