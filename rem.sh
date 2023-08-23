#!/bin/bash

CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > runmining.out &

