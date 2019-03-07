#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=2:00:00
#PBS -l nodes=1:gpus=1
#PBS -l feature=k20
#PBS -o $HOME/histo/data/rm311_128requad/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/data/rm311_128requad/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/.deep_env

cd /home/rpizarro/histo/src
python ribbon.retile_quad.py


