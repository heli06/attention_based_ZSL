#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
##SBATCH --nodelist=ewi1
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/SpeechVisual/Speech_visually_embedding_final

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=./run/Batch_class_1.outputs sh ./run/Batch_class_1.sh