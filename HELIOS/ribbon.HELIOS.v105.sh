#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=3:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/model/ep100/v105/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/model/ep100/v105/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd /home/rpizarro/histo/src
python ribbon.train_unet.model.py 105 100

cd /home/rpizarro/histo/src/HELIOS
# msub ribbon.HELIOS.v105.sh

