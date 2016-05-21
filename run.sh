#!/bin/sh

python3 ./main/fake_config_creator.py config_creator.cfg
mpirun -q -n $1 python3 ./main/electrons.py ../config.cfg ~/result
