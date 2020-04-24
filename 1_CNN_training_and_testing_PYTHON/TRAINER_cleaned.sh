#!/bin/bash
#SBATCH --gres=gpu:1   # requests GPU "generic resource"
#SBATCH --cpus-per-task=8 # max CPU cores per GPU request (6 on Cedar, 16 on Graham).
#SBATCH --mem=64000M   # memory per node (in mbs)
#SBATCH --time=00:15:00 # time (DD-HH:MM) OR (hh:mm:ss) format
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source tensorflow/bin/activate
python ./TRAINER_cleaned.py --variable_update=parameter_server --local_parameter_device=cpu # script to run (change to Myelin UNet)
