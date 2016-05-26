#!/usr/bin/env bash

mpirun -q -n $1 python3 ./main/electrons.py $2 ~/result
python3 ./main/visualizer.py ~/result.npy ~/result.mp4 subprocess
