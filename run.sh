#!/bin/sh

python3 ./main/fake_config_creator.py config_creator.cfg
mpirun -q -n 2 python3 ./main/electrons.py ../config.cfg ~/result
