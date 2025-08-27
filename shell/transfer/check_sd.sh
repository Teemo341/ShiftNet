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

python -u -m training.tool.check \
    --sd_path1 ./models/first_stage_models/kl-f8/model.ckpt \
    --sd_path2 ./models/first_stage_models/kl-f8model_from_prun.ckpt \
