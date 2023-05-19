#!/bin/bash
time1 = `$(date +%Y-%m-%d-%H%M)`
#nohup python -u preprocess.py > sample-inductive${time1}.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-2inductive${time1}.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u knowledge_graph_utils.py > kbtest.out${time1}.out &

CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-rotate-conv${time1}.out &