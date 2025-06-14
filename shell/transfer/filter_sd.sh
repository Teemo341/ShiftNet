#! /bin/bash
#SBATCH -J filter_sd
#SBATCH -o ./shell/results/filter_sd.out
#SBATCH -p compute1
#SBATCH -A compute1         
#SBATCH --qos=compute1             
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m training.tool.remove \
    --controlnet_path ./models/controlnetv1/control_sd15_hed.pth \
    --output_path ./models/sd/v1-5-pruned_hed.ckpt \