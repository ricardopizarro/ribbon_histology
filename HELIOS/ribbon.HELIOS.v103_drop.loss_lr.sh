#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k20
#PBS -o $HOME/histo/weights/lr/v103_drop/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/lr/v103_drop/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.loss_lr.py 103_drop 150

cd /home/rpizarro/histo/src/HELIOS
# msub ribbon.HELIOS.v103_drop.sh

