#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=6:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/prediction/NN_arch/v105_drop/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/prediction/NN_arch/v105_drop/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.test_unet.py 105_drop 100


