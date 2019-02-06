#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=0:30:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/repro/v101_ep010/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/repro/v101_ep010/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.ep010.py 101 10

cd /home/rpizarro/histo/src/HELIOS
# msub ribbon.HELIOS.v101.ep010.sh

