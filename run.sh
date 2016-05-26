#!/bin/sh

mpirun -q -n $1 python3 ./main/electrons.py `python3 ./main/autoconfig.py autoconfig.cfg` ~/result
python3 ./main/visualizer.py ~/result.npy ~/result.mp4 subprocess
