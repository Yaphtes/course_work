#!/usr/bin/env bash

name=result_`date +%y-%m-%d_%H-%M-%S`
mpirun -q -n $1 python3 ./main/electrons.py $2 ~/${name}
python3 ./main/visualizer.py ~/${name}.npy ~/${name}.mp4 subprocess
