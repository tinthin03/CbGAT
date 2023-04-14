#!/bin/bash
time1 = $(date +%Y-%m-%d-%H%M)
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > trainEM-sc-"$time1".out &
CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > samplerules6++-"$time1".out &
