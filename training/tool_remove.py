import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SD weights from a ControlNet or ShiftNet checkpoint.')
    parser.add_argument('--controlnet_path', type=str, required=True, help='Path to the input model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output SD model file.')
    args = parser.parse_args()

    controlnet_path = args.controlnet_path
    output_path = args.output_path

    assert os.path.exists(controlnet_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    checkpoint = torch.load(controlnet_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    for k in list(state_dict.keys()):
        if 'control' in k or 'shift' in k:
            del state_dict[k]
            print(f'Removed {k} from state_dict.')

    torch.save(state_dict, output_path)
    print(f'SD weights saved to {output_path}')