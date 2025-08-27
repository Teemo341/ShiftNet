#! /bin/bash
#SBATCH -J VCA
#SBATCH -o ./results/VCA.out               
#SBATCH -p compute1
#SBATCH -A compute1         
#SBATCH --qos=compute1             
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m training.data.dl_laionart \
    --function download filter \
    --data_dir data/laionart \
    --num_threads 64 \
    --start_idx 0 \
    --end_idx 3274199 \
    --languages en \