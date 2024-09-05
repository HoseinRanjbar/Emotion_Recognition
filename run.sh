#!/bin/bash
### Here are the SBATCH parameters that you should always consider:

### Here are the SBATCH parameters that you should always consider:
#SBATCH --gpus=A100:1
# SBATCH --gpus=1
#SBATCH --time=0-20:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=12
# SBATCH --output=fine_tuning.out


module load mamba
module load gpu
module load cuda
module load python/3.12
source /home/hranjb/data/islr/.env/bin/activate
cd /home/hranjb/data/emotion_recognition

python -u train.py --data /home/hranjb/data/dataset_islr101/Sign_Language_Dataset --num_classes 7 --lookup_table /home/hranjb/data/emotion_reconition/tools/data/emotion_lookup_table.json --recognition 'emotion' --emb_network 'mb2' --save_dir ./output

