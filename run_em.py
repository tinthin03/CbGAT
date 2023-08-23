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
from em_model import EMiner,max_rule_length

DATA_EM_DIR          = "./data_em/umls"
DATA_DIR             = args.data    if len(sys.argv) <= 1 else eval(sys.argv[1])
OUTPUT_DIR           = args.output_folder    if len(sys.argv) <= 2 else eval(sys.argv[2])
write_log_to_console = True         if len(sys.argv) <= 3 else eval(sys.argv[3])
start                = 0            if len(sys.argv) <= 4 else int(sys.argv[4])
hop                  = 1            if len(sys.argv) <= 5 else int(sys.argv[5])
RotatE               = 'RotatE_500' if len(sys.argv) <= 6 else sys.argv[6]

if exp == 'wn18' :RotatE= 'RotatE_200'
if exp == 'umls' :RotatE= 'RotatE_2000'
hyperparams          =  {
    # See model_rnnlogic.py for default values
    'rotate_pretrained': f"{DATA_DIR}/{RotatE}",
    
    # other hyperparameters here
    
}
if len(sys.argv) > 7:
    hyperparams = dict(sys.argv[7])


#DATA_DIR          =args.data #"./data/FB15k-237-direct-pretr" #args.data
#OUTPUT_DIR        =args.output_folder
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
dataset = load_dataset(f"{DATA_DIR}",exp=exp)#键值为['train', 'valid', 'test']的字典，每个字典的key是三元组(h,r,t)，及反三元组(t,r+mov,h)=的list

# Step 2: Generate rules
# Note: This step only needs to do once.
#dataset_graph(dataset, 'train')使用训练集生成KG类Graph
#use_graph，利用graph对象，生成图
#对每个谓词r，抽样若干可能路径作为规则，length_time表示每个三元组各长度规则的抽样数目,对每个r抽样用到num_samples个三元组(默认1000个)
#并评估规则的先验权重

rule_sample.use_graph(dataset_graph(dataset, 'train'))
for r in range(start, dataset['R'], hop):#遍历关系r，抽样其相关的三元组知识,并抽取可能的规则路径
    # Usage: rule_sample.sample(relation, dict: rule_len -> num_per_sample, num_samples, ...)
    num_tr = dataset['T'][int(r)]
    num_samples = 1000+2*num_tr
    #num_samples = 1000
    #print("Predicate",r,"has",num_tr,"tripples,sample rules",num_samples,"times from its tripples.")
    #rules = rule_sample.sample(r, {1: 3, 2: 30, 3: 30, 4: 40, 5: 60, 6: 100}, num_samples, num_threads=12, samples_per_print=100)
    if max_rule_length == 4:
        rules = rule_sample.sample(r, {1: 3, 2: 30, 3: 30, 4: 30}, num_samples, num_threads=12, samples_per_print=100)
    elif max_rule_length == 3:
        rules = rule_sample.sample(r, {1: 3, 2: 30, 3: 30}, num_samples, num_threads=12, samples_per_print=100)
    elif max_rule_length == 2:
        rules = rule_sample.sample(r, {1: 3, 2: 30}, num_samples, num_threads=12, samples_per_print=100)

    print("Sample",len(rules),"rules.")
    rule_sample.save(rules, f"{DATA_DIR}/Rules{str(max_rule_length)}++/rules_{r}.txt")#抽样样本集



# Step 3: Create RNNLogic Model

model = EMiner(dataset, hyperparams, print=print)

for name, param in model.named_parameters():
    model.print(f"Model Parameter: {name} ({param.type()}:{param.size()})")

# Step 4: Train and output test results.
result_stat = []
result_sum = []
result_quality = []

# result_stat = [[2.0110e+04, 6.5849e+02, 3.2559e-01, 2.4557e-01, 3.5783e-01, 4.8474e-01]]#204
# result_sum = [[2.0110e+04, 1.8915e+05, 6.7702e+01, 5.2413e+01, 7.4758e+01, 9.7192e+01]]
# result_stat = [[2.7740e+03,5.4656e+02, 1.6028e-01, 1.2331e-01, 1.6853e-01, 2.2609e-01]]#13
# result_sum = [[2.7740e+03, 6.3728e+03, 3.9989e+00, 2.9433e+00, 4.6969e+00, 5.5385e+00]]

# result_stat = [[1.6959e+04, 3.8754e+02, 3.6434e-01, 2.7531e-01, 4.0365e-01, 5.3832e-01]]#93
# result_sum = [[1.6959e+04, 4.6867e+04, 3.6928e+01, 2.9041e+01, 4.0851e+01, 5.1741e+01]]
# result_stat = [[1.3031e+04, 4.4927e+02, 3.2842e-01, 2.4173e-01, 3.6369e-01, 5.0102e-01]]#47
# result_sum = [[1.3031e+04, 3.3010e+04, 1.6373e+01, 1.2399e+01, 1.8516e+01, 2.3757e+01]]

# result_stat = [[3250.0000,  4.9415, 0.7527, 0.6502, 0.8228, 0.9206]]#106
# result_sum = [[3250.0000,  289.9808,   27.7499,   25.0550,   28.5693,   33.9232]]
# result_quality = [torch.tensor([ 3250.0000, 78.1019, 2627.3936,    3.2190])]

# result_stat = [[1.0500e+04, 7.2235e+02, 3.6930e-01, 2.7400e-01, 4.1172e-01, 5.6097e-01]]#30
# result_sum = [[1.0500e+04, 2.1216e+04, 1.2647e+01, 9.8344e+00, 1.3988e+01, 1.7782e+01]]
# result_quality = [torch.tensor([ 1.0500e+04, 6.0419e+01, 3.4865e+03, 3.0010e+00])]

#result_quality = [torch.tensor([ 43.0000, 170.8139,   0.7245,   0.5698,   0.9318,   0.9768])]
lenth_org = len(result_stat)
umls_no_path = [12,40,41,44,45]
ilpc_no_path = [0,18,34]
ilpc_large_no_path = [5,9,13,42,47,48,64,65,66,67]

dict_tri = {}
empty_test = []
for h,r,t in dataset['test']:
    if r in dict_tri.keys():
        dict_tri[r] += 1
    else:
        dict_tri[r] = 0
ctns = 0
for r,cnt in dict_tri.items():
    if cnt == 0:
        empty_test.append(r)
    ctns += cnt
print("dict_tri:",ctns,dict_tri)
print("empty:",empty_test)

# result_stat = [[1.0180e+04, 2.8588e+04, 1.4286e-01, 1.1699e-01, 1.5524e-01, 1.9217e-01]]#ilpcLG8
# result_sum = [[1.0180e+04, 1.5234e+06, 7.4483e+00, 6.0400e+00, 8.2733e+00, 1.0049e+01]]
# result_quality = [torch.tensor([ 1.0180e+04, 13.4440, 89.3595,  0.7486])]
# start = 68

for r in range(start, dataset['R'], hop):
    if exp == 'umls' and r in umls_no_path: continue # no path    
    if exp == 'ilpc' and r in ilpc_no_path: continue # no path
    if exp == 'ilpc-large' and (r in empty_test or r in ilpc_large_no_path): continue 
    model.train_model(r,
                       rule_file=f"{DATA_DIR}/Rules{str(max_rule_length)}++/rules_{r}.txt",
                      #rule_file=f"{DATA_DIR}/Rules/rules_{r}.txt",#temp XXX
                      model_file=f"{OUTPUT_DIR}/model_{r}.pth")
    result_stat.append(model.result)#result[0]为末轮中所有样本的t_list之和，后面依次为mrr,mr,h1,h3,h10
    result_sum.append(model.result)
    result_quality.append(model.result_quality)
    lenth = len(result_stat)+start-lenth_org#中断重启
    print(lenth,"Result by now(num_of_t_list,mrr,mr,h1,h3,h10):")
    
    res_sum = torch.Tensor(result_sum)
    res_sum = res_sum.sum(0)
    print("SUM,lenth:",res_sum,lenth)
    print("AVG:",res_sum/lenth)

    res = torch.Tensor(result_stat)
    wt = torch.nn.functional.normalize(res[:,0],p=1,dim=0).unsqueeze(-1)
    res_wt = (res[:,1:]*wt).sum(0)
    print("WT_AVG:",res_wt)

    res_quality = torch.stack(result_quality)
    #wt = torch.nn.functional.normalize(res[:,0],p=1,dim=0).unsqueeze(-1)
    res_quality_wt = (res_quality[:,1:]*wt).sum(0)
    print("QUALITY:",res_quality_wt)



# Step 5: Merge results, if (start, hop) == (0, 1)
if (start, hop) == (0, 1):
    calc_result(log_filename)