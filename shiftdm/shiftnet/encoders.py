import torch
import pytorch_lightning as pl
from typing import List, Union, Any

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class single_encoder_base(pl.LightningModule):
    def __init__(self,
                 encode_key: str,
                 ckpt_path=None,
                 ignore_keys=[],
                 *args, **kwargs):
        """encode_key: claim which key in the input dict to encode"""
        super().__init__(*args, **kwargs)
        self.encode_key = encode_key

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def forward(self,x_dict: dict):
        """this dict only contain"""
        x = x_dict[self.encode_key] # bchw

        return x
    
class multi_encoder_base(pl.LightningModule):
    def __init__(self,
                 encode_keys: List[str],
                 encoder_configs: OmegaConf, 
                 encode_weight = None, 
                 ckpt_path=None,
                 ignore_keys=[],
                 *args, **kwargs):
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

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
            z = encoder({key: x_dict[key]})  # call the encoder with the input dict
            encoded.append(z * self.encode_weight[self.encode_keys.index(key)])
        z = sum(encoded) / sum(self.encode_weight)

        return z
    
""" specific encoders """

class fill50k_encoder(single_encoder_base):
    def __init__(self, layers = 4, hidden_dim = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.middel_block = torch.nn.Sequential(
            torch.nn.Conv2d(4, hidden_dim, kernel_size=3, stride=1, padding=1), # from first_stage_model, dim 4
            torch.nn.ReLU(),
            *[torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ) for _ in range(layers)],
            torch.nn.Conv2d(hidden_dim, 4, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x_dict: dict):
        x = super().forward(x_dict)
        z = self.middel_block(x)
        return z