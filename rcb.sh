#!/bin/bash
time1 = `$(date +%Y-%m-%d-%H%M)`
CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCBs${time1}.out &
