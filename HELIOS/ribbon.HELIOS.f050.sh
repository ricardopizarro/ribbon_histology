#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=1:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k20
#PBS -o $HOME/histo/weights/attn/decay_050/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/attn/decay_050/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.train_unet.attn.py 101_drop2 100 50

cd /home/rpizarro/histo/src/HELIOS
# msub ribbon.HELIOS.v101.sh

