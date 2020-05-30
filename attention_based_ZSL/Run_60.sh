#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --workdir=/data/hl/attention_based_ZSL/attention_based_ZSL

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=relation_60.outputs python run2.py --n_epochs 60 --smooth_gamma_r 60.0 --save_file smooth_60.text
