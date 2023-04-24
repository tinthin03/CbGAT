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

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args_wn18():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/WN18RR/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/wn/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

def parse_args():
    args = argparse.ArgumentParser()
    
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/FB15k-237/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/fb/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=10000, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")

    args = args.parse_args()

    return args
args = parse_args()
# %%

inv_relatation = True

def load_data(args):
    #train_data：(train_triples, train_adjacency_mat)，即(三元组id， (rows, cols, data)),其中三元组id的每行为 (entity2id[e1], relation2id[relation], entity2id[e2])
    #其中，train_triples是作为训练数据，而train_adjacency_mat是作为构建图的依据
    ##validation_data未用，test_data不参与图的训练、评估，仅用于最后打分模型conv的评估样本。另外他们组成的三元组全集用于负样本生成、辅助进行conv的评估。
    ##负样本：gat阶段，用于transE的marginloss，而conv阶段，正负样本通过SoftMarginLoss函数生成label和概率得分之间的loss
    #, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
    #    args.data, is_unweigted=False, directed=True)
    file = args.data + "/load_data_pickle.pickle"
    if os.path.exists(file):
        with open(file, 'rb') as handle:
            load_data_pickle = pickle.load(handle)
            train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,unique_cb_entities = load_data_pickle
    else:
        train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,\
            unique_cb_entities = build_cubic_data(args.data, is_unweigted=False, directed=(not inv_relatation))

        load_data_pickle = (train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,unique_cb_entities)

        file = args.data + "/load_data_pickle.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(load_data_pickle, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    # unique_cb_entities_train = []
    # id2cd_entity = {v: k for k, v in relation2id.items()}
    # for rel in unique_cb_entities:
    #     unique_cb_entities_train.append(id2cd_entity[rel])
    # load_data_pickle = (train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
    #         train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,unique_cb_entities_train)
    # file = args.data + "/load_data_pickle-fix.pickle"
    # with open(file, 'wb') as handle:
    #     pickle.dump(load_data_pickle, handle,
    #                 protocol=pickle.HIGHEST_PROTOCOL)
    # unique_cb_entities = unique_cb_entities_train
    #Rnum = len(relation2id)
    if args.pretrained_emb:#vec文件固定为100维
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'),inv = inv_relatation)
        print("Initialised relations and entities from TransE")

    else:# args.embedding_size 默认为50维
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")
    cb_entity_embeddings = relation_embeddings#cb层间图的实体嵌入初始化为层内图的关系的初始值
    #cb_relation_embeddings = entity_embeddings
    cb_relation_embeddings = np.random.randn(len(entity2id), args.embedding_size)#cb层间图的从句关系嵌入初始化为随机值
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    cb_corpus = Corpus(args, train_cb_data, validation_data, test_data, cb_entity2id, cb_relation2id, cb_headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_cb_entities, args.get_2hop,True)
    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings),cb_corpus,torch.FloatTensor(cb_entity_embeddings), torch.FloatTensor(cb_relation_embeddings)

#Corpus_背景知识图谱，entity_embeddings为初始化的嵌入表
file = args.data + "/Corpus_.pickle"
if not os.path.exists(file):
    print("Init datas Generating  >>>")
    Corpus_, entity_embeddings, relation_embeddings,cb_Corpus_, cb_entity_embeddings, cb_relation_embeddings = load_data(args)

    file = args.data + "/Corpus_.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/entity_embeddings.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(entity_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/relation_embeddings.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(relation_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    file = args.data + "/cb_Corpus_.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(cb_Corpus_, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/cb_entity_embeddings.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(cb_entity_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    file = args.data + "/cb_relation_embeddings.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(cb_relation_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    if(args.get_2hop):
        file = args.data + "/2hop.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(Corpus_.node_neighbors_2hop, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

else:
    print("Loading Generated datas  >>>")
    Corpus_ = pickle.load(open(args.data + "/Corpus_.pickle",'rb'))
    entity_embeddings = pickle.load(open(args.data + "/entity_embeddings.pickle",'rb'))
    relation_embeddings = pickle.load(open(args.data + "/relation_embeddings.pickle",'rb'))
    cb_Corpus_ = pickle.load(open(args.data + "/cb_Corpus_.pickle",'rb'))
    cb_entity_embeddings = pickle.load(open(args.data + "/cb_entity_embeddings.pickle",'rb'))
    cb_relation_embeddings = pickle.load(open(args.data + "/cb_relation_embeddings.pickle",'rb'))

if(args.use_2hop): #fb15k中不使用
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

        
#print("\n len(Corpus_.train_indices) ==", len(Corpus_.train_indices))
#print("\n len(cb_Corpus_.train_indices) ==", len(cb_Corpus_.train_indices))

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)
cb_entity_embeddings_copied = deepcopy(cb_entity_embeddings)
cb_relation_embeddings_copied = deepcopy(cb_relation_embeddings)
print("Initial cb_entity dimensions {} , cb_relation dimensions {}".format(
    cb_entity_embeddings.size(), cb_relation_embeddings.size()))#当纯随机时，是50维（args决定）。加入pretrain时，cb_entity_embeddings为100维，cb_relation_embeddings依然由args决定
print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))#当纯随机时，都是50维（args决定）。但加入pretrain时，为100维。
# %%

CUDA = torch.cuda.is_available()

#采用transE的目标函数计算loss，训练GAT
#train_indices，训练集的id索引，每行是一个三元组
def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed,mod = 0):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))#获取正负样本的分界线
    print("train_indices.shape",train_indices.shape)
    print("len_pos_triples",len_pos_triples)

    pos_triples = train_indices[:len_pos_triples] #前一部分是正样本,长3333
    if mod == 0:
        neg_triples = train_indices[len_pos_triples:]
    else:
        neg_triples = train_indices[len_pos_triples:]
    #print("pos_triples.shape",pos_triples.shape)
    #print("neg_triples.shape",neg_triples.shape)

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)
    #print("pos_triples.shape",pos_triples.shape)#6666

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()
    # print("pos_norm.shape",pos_norm.shape)#6666
    # print("neg_norm.shape",neg_norm.shape)#

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss

#parallel
device_ids=[0,1]
#out_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def train_gat(args):
    print("train_gat ...")
    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT
    #cb_model_gat = SpKBGATModified(cb_entity_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,args.drop_GAT, args.alpha, args.nheads_GAT) #cubic GAT

    #print("entity_embeddings.shape,relation_embeddings.shape",entity_embeddings.shape,relation_embeddings.shape)
    if CUDA:
        model_gat.cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
    
    #model_gat.to(out_device)

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([]).long()
    #Corpus_为一个Corpus对象
    #node_neighbors_2hop（get_further_neighbors函数返回值）：表示邻居节点的字典，
    # 形式为，neighbors[source][distance]=[],列表中形为(tuple(relations), tuple(entities[:-1]))
    #current_batch_2hop_indices为当前batch实体的2hop邻居实体，list形式，其中元素为一个2hop路径e1,relations[-1],relations[0],e2
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32) ##转为int格式的训练三元组id数据

        model_gat.train()  # getting in training mode，这里仅是切换为train模式
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1
        print("\n num_iters_per_epoch ==", num_iters_per_epoch)
        print("\n len(Corpus_.train_indices) ==", len(Corpus_.train_indices))


        for iters in range(num_iters_per_epoch):
            print("\n iters-> ", iters)
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)#循环取出第iter_num个batch,同时生成正负样本，label分别是1、-1

            print("\n len(train_indices) ==", len(train_indices))
            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            #train_adj_matrix为(adj_indices, adj_values)，每行是([e1,e2],r),r为关系id或者1
            #train_indices为转为int格式的训练三元组id数据
            entity_embed, relation_embed,entity_l_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_gat, args.data, epoch,
                   args.output_folder)


def train_gat_cb(args):
    print("train_gat_cb ...")
    # Creating the gat model here.
    ####################################

    print("Load model")
    #epoch_load = args.epochs_gat #默认取模型最后一个epcoh
    epoch_load = 0#默认初始化训练时
    epoch_load = 0#载入之前的模型时。手动指定载入的模型epoch,注意是文件名里的数字+1

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT

    #print("train_gat_cb entity_embeddings.shape,relation_embeddings.shape",entity_embeddings.shape,relation_embeddings.shape)

    #model_gat.to(out_device)
    pre  = 'cb_e'
    if epoch_load>0:
        model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_e{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    final_entity_embeddings = model_gat.final_entity_embeddings
    final_relation_embeddings = model_gat.final_relation_embeddings#得到GAT模型的最终嵌入

    #model_gat = SpKBGATModified(final_entity_embeddings, final_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    #                            args.drop_GAT, args.alpha, args.nheads_GAT) #GAT
    
    cb_model_gat = SpKBGATModified(model_gat.relation_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    args.drop_GAT, args.alpha, args.nheads_GAT,cb_flag=True) #cubic GAT

    pre  = 'cb_r'
    if epoch_load>0:
        cb_model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_r{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    cb_final_entity_embeddings = cb_model_gat.final_entity_embeddings
    cb_final_relation_embeddings = cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入

    #print("train_gat_cb final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape",final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape)
     
    if CUDA:
        model_gat.cuda()
        cb_model_gat.cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
            cb_model_gat = nn.DataParallel(cb_model_gat, device_ids=[torch.cuda.current_device()])

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    cb_optimizer = torch.optim.Adam(
        cb_model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    #test 参数重复
    # param_set = set()#一个generator objects，只能调用一次。
    # param_groups =[{'params': model_gat.parameters()}] 
    # param_group = {'params': cb_model_gat.parameters()}
    # for group in param_groups:
    #     #print(" model_gat parameters():",list(group['params']))
    #     param_set.update(set(group['params']))
    # #print(" cb_model_gat parameters():",list(param_group['params']))
    # #print(" cb_model_gat parameters():\n",list(param_group['params']))
    # if not param_set.isdisjoint(set(param_group['params'])):
    #     print("some parameters appear in more than one parameter group")

    indiv_params =  []
    idx = 0
    for p in cb_model_gat.parameters():
        if idx!=2:indiv_params += [p]
        idx += 1     
    all_optimizer = torch.optim.Adam([{'params': model_gat.parameters()},
                             {'params': indiv_params}],
                             lr=args.lr, weight_decay=args.weight_decay_gat)

    #print("model_gat.parameters(),cb_model_gat.parameters()",model_gat.parameters(),cb_model_gat.parameters())
    #print("list model_gat.parameters(),cb_model_gat.parameters()",list(model_gat.named_parameters()),list(cb_model_gat.named_parameters()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    cb_scheduler = torch.optim.lr_scheduler.StepLR(
        cb_optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    all_scheduler = torch.optim.lr_scheduler.StepLR(
        all_optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([]).long()
    #Corpus_为一个Corpus对象
    #node_neighbors_2hop（get_further_neighbors函数返回值）：表示邻居节点的字典，
    # 形式为，neighbors[source][distance]=[],列表中形为(tuple(relations), tuple(entities[:-1]))
    #current_batch_2hop_indices为当前batch实体的2hop邻居实体，list形式，其中元素为一个2hop路径e1,relations[-1],relations[0],e2
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Loaded epochs {}".format(epoch_load))
    print("Number of epochs {}".format(args.epochs_gat))

    init_flag = True
    cb_entity_embed = cb_entity_embeddings #仅为了避免这个变量在model_gat第一次调用时的提前赋值语法错误
    #cb_relation_embed = cb_relation_embeddings
    for epoch in range(epoch_load,args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        random.shuffle(cb_Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32) ##转为int格式的训练三元组id数据
        cb_Corpus_.train_indices = np.array(
            list(cb_Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode，这里仅是切换为train模式
        cb_model_gat.train()
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1
        print("\n num_iters_per_epoch ==", num_iters_per_epoch)
        print("\n len(Corpus_.train_indices) ==", len(Corpus_.train_indices))
        print("\n len(cb_Corpus_.train_indices) ==", len(cb_Corpus_.train_indices))

        for iters in range(num_iters_per_epoch):
            print("\n iters-> ", iters)
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)#循环取出第iter_num个batch,同时生成正负样本，label分别是1、-1
            cb_train_indices, cb_train_values = cb_Corpus_.get_iteration_batch(iters) #数量应该不一致，主要是负样本数量不同
            print("\n len(train_indices) ==", len(train_indices))
            print("\n len(cb_train_indices) ==", len(cb_train_indices))
            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
                cb_train_indices = Variable(
                    torch.LongTensor(cb_train_indices)).cuda()
                cb_train_values = Variable(torch.FloatTensor(cb_train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))
                cb_train_indices = Variable(torch.LongTensor(cb_train_indices))
                cb_train_values = Variable(torch.FloatTensor(cb_train_values))

            # forward pass
            #train_adj_matrix为(adj_indices, adj_values)，每行是([e1,e2],r),r为关系id或者1
            #train_indices为转为int格式的训练三元组id数据
            print("\n Run model_gat forward",)
            if init_flag:
                entity_embed, relation_embed,entity_l_embed = model_gat(
                    Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)
                init_flag = False
            else:
                entity_embed, relation_embed,entity_l_embed = model_gat(
                    Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices,ass_rel=cb_entity_embed)

            
            #optimizer.zero_grad()

            #loss = batch_gat_loss(
            #    gat_loss_func, train_indices, entity_embed, relation_embed)

            # loss.backward()
            # optimizer.step()

            # epoch_loss.append(loss.data.item())

            print("\n Run cb_model_gat forward",)
            
            cb_entity_embed, cb_relation_embed,cb_entity_l_embed = cb_model_gat(
                cb_Corpus_, cb_Corpus_.train_adj_matrix, cb_train_indices, current_batch_2hop_indices,ass_ent=relation_embed)

            #cb_optimizer.zero_grad()

            # cb_loss = batch_gat_loss(
            #     gat_loss_func, cb_train_indices, cb_entity_embed, cb_relation_embed,mod = 1)

            # cb_loss.backward()
            # cb_optimizer.step()

            # epoch_loss.append(cb_loss.data.item())
            
            all_optimizer.zero_grad()
            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)
            cb_loss = batch_gat_loss(
                gat_loss_func, cb_train_indices, cb_entity_embed, cb_relation_embed,mod = 1)

            all_loss = loss+cb_loss
            all_loss.backward()
            all_optimizer.step()

            epoch_loss.append(all_loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        # scheduler.step()
        # cb_scheduler.step()
        all_scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch%100==0 or epoch == args.epochs_gat - 1:
            save_model(model_gat, args.data, epoch,
                    args.output_folder,prex = 'cb_e')
            save_model(cb_model_gat, args.data, epoch,
                   args.output_folder,prex = 'cb_r')

#使用训练集Corpus_.train_triples训练打分函数model_conv
def train_conv(args):
    print("train_conv ...")#

    # Creating convolution model here.
    ####################################

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()
    prex = 'cb_e'
    model_gat.load_state_dict(torch.load(
        '{}/trained_{}{}.pth'.format(args.output_folder,prex, args.epochs_gat - 1)), strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings#得到GAT模型的最终嵌入

    Corpus_.batch_size = args.batch_size_conv #修改Corpus_的batch_size
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv) #设置conv训练过程的负样本比例

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss() #用于二分类问题(label1、-1)的loss，预测输出为一个对1/-1label的概率值

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode，切换为train模式，基于GAT的嵌入向量输入，训练convKB
        start_time = time.time()
        epoch_loss = []
        #每个batch的长度batch_size_conv，计算总共需要多少个batch/iter来跑完训练
        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):#每个iter实际上是一个batch，Corpus_的batch_size已改为args.batch_size_conv
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)#循环取出第iter_num个batch,同时生成正负样本，label分别是1、-1

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices) #ConvKB输出预测结果

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_conv, args.data, epoch,
                   args.output_folder + "conv/")

#全部模型训练完毕后的验证过程
def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities) #验证计算结果的过程,仅此函数调用了self.test_indices，无返回值，直接统计结果


import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


def cleanup_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if "module" in k:
            new_name = k[7:]
        else:
            new_name = k
        new_state[new_name] = v
    return new_state

if __name__ == "__main__":
    print("main0.py init complete")
    print("train_gat & cubic ...")
    #train_gat(args)
    #print("train_gat complete")
    train_gat_cb(args)
    print("train_gat_cb complete")
    train_conv(args)
    evaluate_conv(args, Corpus_.unique_entities_train)
