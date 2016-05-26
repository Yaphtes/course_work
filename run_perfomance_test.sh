#!/usr/bin/env bash

CFG_FILE=`python3 ./main/autoconfig.py autoconfig.cfg`

for i in `seq 1 $1`;
do
    mpirun -q -n ${i} python3 ./main/electrons.py ${CFG_FILE} -testperf
done