import sys
import os

import torch
from omegaconf import OmegaConf
import argparse
from share import *
from shiftdm.model import create_model
from ldm.util import instantiate_from_config, exists

#! I strongly recommend using a pretrained shift encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add shift encoder to a pre-trained model.')
    parser.add_argument('--config_path', type=str, default='./models/shiftdm_v15.yaml', help='Path to the model configuration file.')
    parser.add_argument('--sd_path', type=str, help='Path to the input model file.')
    parser.add_argument('--shift_path', type=str, default=None, help='Path to the shift encoder weights file.')
    parser.add_argument('--output_path', type=str, help='Path to save the output model file.')
    args = parser.parse_args()

    input_path = args.sd_path
    if args.shift_path is not None:
        shift_path = args.shift_path
    output_path = args.output_path

    assert os.path.exists(input_path), 'Input model does not exist.'
    if shift_path is not None:
        assert os.path.exists(shift_path), 'Shift encoder weights file does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    model = create_model(config_path='./models/cldm_v15.yaml')

    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    model.load_state_dict(pretrained_weights, strict=False) # load without shift encoder
    if shift_path is not None:
        shift_weights = torch.load(shift_path)
        if 'state_dict' in shift_weights:
            shift_weights = shift_weights['state_dict']
        base_config = OmegaConf.load(args.config_path)
        shift_stage_config = base_config.shift_stage_config
        shift_stage_model = instantiate_from_config(shift_stage_config)
        shift_stage_model.load_state_dict(shift_weights, strict=False)
        model.shift_stage_model = shift_stage_model
        if exists(model.shift_stage_model.decoder):
            model.shift_stage_model.decoder = None  # remove decoder to save memory, only need the encoder


    torch.save(model.state_dict(), output_path)
    print('Done.')
