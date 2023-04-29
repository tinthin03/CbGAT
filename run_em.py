import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data,build_cubic_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle

from main import *

from merge_results import calc_result
import rule_sample
from knowledge_graph_utils import *
#from groundings import *
from em_model import EMiner


DATA_EM_DIR          = "./data_em/FB15k-237"    if len(sys.argv) <= 1 else eval(sys.argv[1])
OUTPUT_DIR           = "./checkpoints/fb/em_out3"    if len(sys.argv) <= 2 else eval(sys.argv[2])
write_log_to_console = True         if len(sys.argv) <= 3 else eval(sys.argv[3])
start                = 0            if len(sys.argv) <= 4 else int(sys.argv[4])
hop                  = 1            if len(sys.argv) <= 5 else int(sys.argv[5])
RotatE               = 'RotatE_500' if len(sys.argv) <= 6 else sys.argv[6]
hyperparams          =  {
    # See model_rnnlogic.py for default values
    'rotate_pretrained': f"{DATA_EM_DIR}/{RotatE}",
    
    # other hyperparameters here
    
}
if len(sys.argv) > 7:
    hyperparams = dict(sys.argv[7])

DATA_DIR          = "./data/FB15k-237-direct-pretr" 
'''
Example:
The script will train a separate model for each relation in range(start, total number of relations, hop).

DATA_DIR             = "../data/kinship"
OUTPUT_DIR           = "./checkpoints/"
write_log_to_console = True
start                = 0
hop                  = 1
RotatE               = 'RotatE_500'
'''

old_print = print
log_filename = f"{OUTPUT_DIR}/train_log.txt"
log_file = open(log_filename, 'a')

def new_print(*args, **kwargs):
    if write_log_to_console:
        old_print(*args, **kwargs, flush=True)
    old_print(*args, **kwargs, file=log_file, flush=True)

print = new_print

# Step 0: Install dependencies
# os.chdir('./cppext')
# os.popen('python setup.py install')
# os.chdir('..')

# Step 1: Load dataset
dataset = load_dataset(f"{DATA_DIR}")#键值为['train', 'valid', 'test']的字典，每个字典的key是三元组(h,r,t)，及反三元组(t,r+mov,h)=的list

# Step 2: Generate rules
# Note: This step only needs to do once.
#dataset_graph(dataset, 'train')使用训练集生成KG类Graph
#use_graph，利用graph对象，生成图
#对每个谓词r，抽样若干可能路径作为规则，length_time表示每个三元组各长度规则的抽样数目,对每个r抽样用到num_samples个三元组(默认1000个)
#并评估规则的先验权重

# rule_sample.use_graph(dataset_graph(dataset, 'train'))
# for r in range(start, dataset['R'], hop):#遍历关系r，抽样其相关的三元组知识,并抽取可能的规则路径
#     # Usage: rule_sample.sample(relation, dict: rule_len -> num_per_sample, num_samples, ...)
#     # num_tr = dataset['T'][int(r)]
#     # num_samples = 1000+2*num_tr
#     num_samples = 1000
#     #print("Predicate",r,"has",num_tr,"tripples,sample rules",num_samples,"times from its tripples.")
#     #rules = rule_sample.sample(r, {1: 3, 2: 30, 3: 30, 4: 40, 5: 60, 6: 100}, num_samples, num_threads=12, samples_per_print=100)
#     rules = rule_sample.sample(r, {1: 3, 2: 30, 3: 30, 4: 30}, num_samples, num_threads=12, samples_per_print=100)
#     print("Sample",len(rules),"rules.")
#     rule_sample.save(rules, f"{DATA_DIR}/Rules/rules_{r}.txt")#抽样样本集



# Step 3: Create RNNLogic Model

model = EMiner(dataset, hyperparams, print=print)

for name, param in model.named_parameters():
    model.print(f"Model Parameter: {name} ({param.type()}:{param.size()})")

# Step 4: Train and output test results.
result_stat = []
for r in range(start, dataset['R'], hop):
    model.train_model(r,
                      rule_file=f"{DATA_DIR}/Rules4++/rules_{r}.txt",
                      model_file=f"{OUTPUT_DIR}/model_{r}.pth")
    result_stat.append(model.result)#result[0]为末轮中所有样本的t_list之和，后面依次为mrr,mr,h1,h3,h10
    lenth = len(result_stat)
    print(lenth,"Result by now(num_of_t_list,mrr,mr,h1,h3,h10):")
    res = torch.Tensor(result_stat)
    res_sum = res.sum(0)
    print("SUM,lenth:",res_sum,lenth)
    print("AVG:",res_sum/lenth)
    wt = torch.nn.functional.normalize(res[:,0],p=1,dim=0).unsqueeze(-1)
    res_wt = (res[:,1:]*wt).sum(0)
    print("WT_AVG:",res_wt)



# Step 5: Merge results, if (start, hop) == (0, 1)
if (start, hop) == (0, 1):
    calc_result(log_filename)