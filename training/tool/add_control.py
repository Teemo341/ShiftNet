import sys
import os

import torch
import argparse
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add control nodes to a pre-trained model.')
    parser.add_argument('--config_path', type=str, default='./models/controlnet/cldm_v15.yaml', help='Path to the model configuration file.')
    parser.add_argument('--sd_path', type=str, help='Path to the input model file.')
    parser.add_argument('--output_path', type=str, help='Path to save the output model file.')
    args = parser.parse_args()

    input_path = args.sd_path
    output_path = args.output_path

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.


    model = create_model(config_path=args.config_path)

    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')
