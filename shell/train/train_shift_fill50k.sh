#! /bin/bash
#SBATCH -J Shiftnet
#SBATCH -o ./results/Shiftnet_fill50k.out               
#SBATCH -p compute1
#SBATCH -A compute1         
#SBATCH --qos=compute1             
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m training.main  --base models/shiftdm/shift_sd15.yaml --train