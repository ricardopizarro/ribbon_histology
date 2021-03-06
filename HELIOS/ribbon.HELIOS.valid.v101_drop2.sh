#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=6:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/prediction/NN_arch/valid_10000epochs/v101_drop2/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/prediction/NN_arch/valid_10000epochs/v101_drop2/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.valid_unet.py 101_drop2 100


