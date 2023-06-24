#!/bin/bash
time1 = `$(date +%Y-%m-%d-%H%M)`
#nohup python -u preprocess.py > sample-inductive${time1}.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-2inductive${time1}.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u knowledge_graph_utils.py > kbtest.out${time1}.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-inv200-prl-rotate-conv${time1}.out &
# CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-reverse-rotate-conv${time1}.out &
CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > trainCB-rotate-dim100-conv${time1}.out &


#CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > wnCB-rotate-conv2.2${time1}.out &


# CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > umls-CB-transE-conv${time1}.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u main.py > ilpcLG-CB2-rotate-conv${time1}.out &