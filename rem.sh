#!/bin/bash
time1 = $(date +%Y-%m-%d-%H%M)
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > trainEM-cbaug4++-em5-"$time1".out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > samplerules6++-"$time1".out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > trainEM-meandist2-trgen5.4-noinv-rely_gen-"$time1".out &
CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > trainEM-scgnd-trgen5.4-noinv-rely_gen-"$time1".out &
