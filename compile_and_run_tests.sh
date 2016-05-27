#!/usr/bin/env bash

python3 -m compileall -qq -b ./main

CFG_FILE=`python3 ./main/autoconfig.pyc autoconfig.cfg`

for i in `seq 1 $1`;
do
    mpirun -q -n ${i} python3 ./main/electrons.pyc ${CFG_FILE} -testperf
done