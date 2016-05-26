#!/bin/sh

python3 ./main/autoconfig.py config_creator.cfg
mpirun -q -n $1 python3 ./main/electrons.py ../config.cfg ~/result
python3 ./main/visualizer.py ~/result.npy ~/result.mp4 subprocess
