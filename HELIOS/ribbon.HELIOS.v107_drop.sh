#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=4:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/NN_arch/v107_drop/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/NN_arch/v107_drop/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.py 107_drop 100

cd /home/rpizarro/histo/src/HELIOS
# msub ribbon.HELIOS.v107_drop.sh

