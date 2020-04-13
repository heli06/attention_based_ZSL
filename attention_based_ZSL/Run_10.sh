#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/MSc/attention_based_ZSL/attention_based_ZSL

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=relation.outputs python run2.py --smooth_gamma_r 10.0 --save_file smooth_10.text
