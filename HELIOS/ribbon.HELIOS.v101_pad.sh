#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/NN_arch/v101_pad/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/NN_arch/v101_pad/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.py 101_pad 100

cd /home/rpizarro/histo/src/HELIOS
msub ribbon.HELIOS.v101_pad.sh

