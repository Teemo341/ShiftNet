import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class single_encoder_base(pl.LightningModule):
    def __init__(self, encode_key: str, *args, **kwargs):
        """encode_key: claim which key in the input dict to encode"""
        super().__init__(*args, **kwargs)
        self.encode_key = encode_key

    def forward(self,x_dict: dict):
        """this dict only contain"""
        x = x_dict[self.encode_key] # bchw

        return x
    
class multi_encoder_base(pl.LightningModule):
    def __init__(self, encode_keys:list[str], encoder_configs:OmegaConf, encode_weight = None, *args, **kwargs):
        """
        Args:
            encode_key:claim each encoder should encode which key
            encoder_configs: OmegaConf object containing configurations for each encoder
            encode_weight: a list of weights for each encoder, defaults none means equal
        """
        super().__init__(*args, **kwargs)
        self.encode_keys = encode_keys
        self.encoders = torch.nn.ModuleList()
        self.encode_weight = encode_weight if encode_weight is not None else [1.0] * len(encode_keys)
        for key, config in zip(encode_keys, encoder_configs):
            encoder = instantiate_from_config(config)
            self.encoders.append(encoder)
            setattr(self, key + "_encoder", encoder)
        assert len(self.encode_keys) == len(self.encoders), "encode_keys and encoders must have the same length"

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

    def forward(self, x_dict: dict):
        """
        Args:
            x_dict: a dict containing the input data, each key should match encode_keys
        Returns:
            z: the weighted average of encoded features from each encoder
        """
        encoded = []
        for key, encoder in zip(self.encode_keys, self.encoders):
            x = x_dict[key] # bchw
            z = encoder(x)
            encoded.append(z * self.encode_weight[self.encode_keys.index(key)])
        z = sum(encoded) / sum(self.encode_weight)

        return z