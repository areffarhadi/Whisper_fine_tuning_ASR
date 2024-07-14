#!/bin/bash

### Here are the SBATCH parameters that you should always consider:
#SBATCH --gpus=A100:1
#SBATCH --time=0-12:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=128G    
#SBATCH --output=whisper_FT.out


### activeate the venv
source /home/arfarh/scratch/Aref_tools/whisper-venv/bin/activate

module load gpu
# module load cudnn
module load cuda/11.8.0


python3 ./prepare_data.py
python3 ./train.py
python3 ./decode.py
