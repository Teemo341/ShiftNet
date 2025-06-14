import os
import torch
import argparse

#! i already checked, all controlnetv1 unet models have different output blocks.
#! so, you should download sd ckpt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main, i'm not sure whether the link still works, anyway you can find it in the diffusers repo.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ckeck if 2 sd weights are same.')
    parser.add_argument('--sd_path1', type=str, required=True, help='Path to the first SD model file.')
    parser.add_argument('--sd_path2', type=str, required=True, help='Path to the second SD model file.')
    args = parser.parse_args()

    sd_path1 = args.sd_path1
    sd_path2 = args.sd_path2
    assert os.path.exists(sd_path1), 'First SD model does not exist.'
    assert os.path.exists(sd_path2), 'Second SD model does not exist.'

    sd1 = torch.load(sd_path1, map_location='cpu')
    sd2 = torch.load(sd_path2, map_location='cpu')
    if 'state_dict' in sd1:
        sd1 = sd1['state_dict']
    if 'state_dict' in sd2:
        sd2 = sd2['state_dict']
        
    missing_keys = 'missing_keys:\n'
    same_keys = 'same_keys:\n'
    different_keys = 'different_keys:\n'

    for k in sd1.keys():
        if k not in sd2:
            missing_keys += f'  {k}\n'
        elif torch.equal(sd1[k], sd2[k]):
            same_keys += f'  {k}\n'
        else:
            different_keys += f'  {k}\n'

    print(missing_keys)
    print(different_keys)
    print(same_keys)
    print('Check completed.')