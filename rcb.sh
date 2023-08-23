#!/bin/bash

CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB.out &
