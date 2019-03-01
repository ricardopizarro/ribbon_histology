#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=3:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/drop/rate_025/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/drop/rate_025/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.drop.py 101 100 25

cd /home/rpizarro/histo/src/HELIOS
msub ribbon.HELIOS.r025.sh

