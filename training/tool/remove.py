import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove checkpoint weights with key.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output SD model file.')
    parser.add_argument('--key', type=str, default='control', help='Key to remove from the state_dict (default: "control").')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    key_to_remove = args.key

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    checkpoint = torch.load(input_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    for k in list(state_dict.keys()):
        if key_to_remove in k:
            del state_dict[k]
            print(f'Removed {k} from state_dict.')

    print(f'Removed {key_to_remove} from {len(state_dict)} keys in state_dict.\n')
    print(f'Keys remaining in state_dict after removal:')
    for k in list(state_dict.keys()):
        print(f' - {k}')

    torch.save(state_dict, output_path)
    print(f'SD weights saved to {output_path}')