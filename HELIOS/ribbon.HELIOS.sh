#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=0:20:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k20
#PBS -o $HOME/histo/prediction/NN_arch/v103_long/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/prediction/NN_arch/v103_long/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.test_unet.long.py v103 94 100


