#!/bin/bash
time1=$(date)
CUDA_VISIBLE_DEVICES="$@" python -u main.py > trainCbGAT${time1}.out
