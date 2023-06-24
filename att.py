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


#import pickle

def generate_id_map(entity2id_file, fb2w_file, id_map_file,e_map_file):
    # 加载实体序号到实体id的映射文件
    entity2id = {}
    with open(entity2id_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            entity2id[parts[0].replace("/m/","")] = parts[1]
    # print("len(entity2id)",len(entity2id))
    # print("/m/04nrcg 's id = ",entity2id["04nrcg"]) #=10

    # 加载freebase实体id到维基百科id的映射文件
    fb2w = {}
    with open(fb2w_file, "r", encoding="utf-8") as f:
        lineind = 0
        for line in f:
            lineind += 1
            if lineind <4:
                continue
            parts = line.strip().split("	")
            if lineind%100000==0:print(parts)
            if len(parts)<3:continue
            if len(parts[2])<2 or len(parts[0])<2:continue
            fb_id = parts[0][1:-1].replace("http://rdf.freebase.com/ns/m.","")
            wiki_id = parts[2][1:-3].replace("http://www.wikidata.org/entity/Q","")
            fb2w[fb_id] = wiki_id

    # 生成id映射的字典，并将其序列化并存储在硬盘上
    id_map = {}
    e_map = {}
    for entity,ids  in entity2id.items():
        #fb_id = "http://rdf.freebase.com/ns/" + entity_id
        wiki_id = fb2w.get(entity)
        if wiki_id:
            id_map[ids] = wiki_id
            e_map[entity] = wiki_id
    with open(id_map_file, "wb") as f:
        pickle.dump(id_map, f)
    with open(e_map_file, "wb") as f:
        pickle.dump(e_map, f)

    return id_map,e_map

def eid2wid(entity_num, id_map_file = args.data + "/eid2wid.pickle"):
    # 加载id映射字典
    with open(id_map_file, "rb") as f:
        id_map = pickle.load(f)

    # 查找对应实体id的维基百科id
    entity_id = str(entity_num)
    wiki_id = id_map.get(entity_id)

    return wiki_id
def e2wid(entity, id_map_file = args.data + "/e2wid.pickle"):
    # 加载id映射字典
    with open(id_map_file, "rb") as f:
        id_map = pickle.load(f)

    # 查找对应实体id的维基百科id
    entity_id = str(entity)
    wiki_id = id_map.get(entity_id)

    return wiki_id

def wiki():
    file_eid = args.data + "/eid2wid.pickle"
    file_e = args.data + "/e2wid.pickle"
    if not os.path.exists(file_eid):
        eid2wid_map,e2wid_map = generate_id_map(args.data +"entity2id.txt",args.data +"fb2w.nt",file_eid,file_e)
        with open(file_eid, 'wb') as handle:
            pickle.dump(eid2wid_map, handle,
                        protocol=pickle.HIGHEST_PROTOCOL) 
        with open(file_e, 'wb') as handle:
            pickle.dump(e2wid_map, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated eid2wid,e2wid  >>>")
        eid2wid_map = pickle.load(open(file_eid,'rb'))
        e2wid_map = pickle.load(open(file_e,'rb'))

#print(e2wid('02vqpx8'))
#print(e2wid('05gp3x'))
#print(eid2wid(1))
    
def read_cd_gat(args):
    print("Load model")
    #epoch_load = args.epochs_gat #默认取模型最后一个epcoh
    epoch_load = 0#默认初始化训练时
    epoch_load = 3000#载入之前的模型时。手动指定载入的模型epoch,注意是文件名里的数字+1

    #print("train_gat_cb entity_embeddings",entity_embeddings)
    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT

    #print("train_gat_cb entity_embeddings.shape,relation_embeddings.shape",entity_embeddings.shape,relation_embeddings.shape)
    #model_gat.to(out_device)
    #print("init final_entity_embeddings...",model_gat.final_entity_embeddings)

    pre  = 'cb_e'
    if epoch_load>0:
        model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_e{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    #print(model_gat.state_dict())

    #model_gat = SpKBGATModified(final_entity_embeddings, final_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    #                            args.drop_GAT, args.alpha, args.nheads_GAT) #GAT
    
    #print("train_gat_cb cb_relation_embeddings",cb_relation_embeddings)
    cb_model_gat = SpKBGATModified(model_gat.relation_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    args.drop_GAT, args.alpha, args.nheads_GAT,cb_flag=True) #cubic GAT

    #print("init cb_final_relation_embeddings...",cb_model_gat.final_relation_embeddings)


    pre  = 'cb_r'
    if epoch_load>0:
        cb_model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_r{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)

    #print("train_gat_cb final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape",final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape)
    
    # if CUDA:
    #     model_gat.cuda()
    #     cb_model_gat.cuda()
    #     if torch.cuda.device_count() > 1:
    #         print("Use", torch.cuda.device_count(), 'gpus')
    #         model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
    #         cb_model_gat = nn.DataParallel(cb_model_gat, device_ids=[torch.cuda.current_device()])

    return model_gat,cb_model_gat

def ShortestPath(args):

    model_gat,cb_model_gat = read_cd_gat(args)
    final_entity_embeddings = model_gat.final_entity_embeddings
    final_relation_embeddings = model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
    cb_final_entity_embeddings = cb_model_gat.final_entity_embeddings
    cb_final_relation_embeddings = cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入

    print("final_entity_embeddings.shape",final_entity_embeddings.shape)
    print("inal_relation_embeddings.shape",final_relation_embeddings.shape)
    print("cb_final_entity_embeddings.shape",cb_final_entity_embeddings.shape)
    print("cb_final_relation_embeddings.shape",cb_final_relation_embeddings.shape)
    # print("final_entity_embeddings...",final_entity_embeddings)
    # print("cb_final_relation_embeddings...",cb_final_relation_embeddings)
    count = cb_final_relation_embeddings.shape[0]
    cos = torch.nn.CosineSimilarity(dim=0,eps=1e-12)#eps小一些，可以体高精度
    # for i in range(count):
    #     if i%100 == 0:
    #         e = final_entity_embeddings[i]
    #         cb_r = cb_final_relation_embeddings[i]
    #         #print((cb_final_relation_embeddings[i]-final_entity_embeddings[i]))
    #         # minus = cb_r-e#1范数接近200-230，2范数接近20
    #         # #minus = cb_r+e#1范数接近200-230，2范数接近20
    #         # nor = torch.norm(minus,p=2,dim=0)
    #         # print(i,nor)

    #         #print(i,e)
    #         #print(i,cb_r)
    #         #output = cos(e,cb_r*(-1))
    #         output = cos(e,cb_r)
    #         angs = (torch.acos(output)*180/3.1415926).item()
    #         output = cos(e,cb_r*(-1))
    #         angs_ = (torch.acos(output)*180/3.1415926).item()

    #         #print(i,angs,angs_,output)
    
    e = final_entity_embeddings
    r = final_relation_embeddings
    cb_e= cb_final_entity_embeddings
    cb_r = cb_final_relation_embeddings
    # e = model_gat.entity_embeddings
    # r = model_gat.relation_embeddings
    # cb_e= cb_model_gat.entity_embeddings
    # cb_r = cb_model_gat.relation_embeddings
    #r[48] = r[66]+cb_r[701]+r[133]
    # check = r[66]+cb_r[10957]+r[133]-r[48]
    # print(r[48])
    # print(check.shape,check)
    # nor = torch.norm(check,p=2,dim=0)
    # print(nor)
    
    # output = cos(r[66]+cb_r[10957]+r[133],r[48])
    # angs = (torch.acos(output)*180/3.1415926).item()
    # print("angs=",angs)
    #(self, args, train_data, validation_data, test_data, entity2id,relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False,cubic = False)
    #cb_corpus = Corpus(cb_Corpus_.,)
    #corpus = Corpus(Corpus_)
    
    file = args.data + "/cb_Corpus_graph.pickle"
    if not os.path.exists(file):
        cb_kg = cb_Corpus_.get_multiroute_graph()
        file = args.data + "/cb_Corpus_graph.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(cb_kg, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        kg = Corpus_.get_multiroute_graph()
        file = args.data + "/Corpus_graph.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(kg, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated graph  >>>")
        cb_kg = pickle.load(open(args.data + "/cb_Corpus_graph.pickle",'rb'))
        kg = pickle.load(open(args.data + "/Corpus_graph.pickle",'rb'))
    file = args.data + "/cb_path_graph.pickle"
    if not os.path.exists(file):
        cb_path_kg = cb_Corpus_.get_path_graph()
        file = args.data + "/cb_path_graph.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(cb_path_kg, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        path_kg = Corpus_.get_path_graph()
        file = args.data + "/path_graph.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(path_kg, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated path_graph  >>>")
        cb_path_kg = pickle.load(open(args.data + "/cb_path_graph.pickle",'rb'))
        path_kg = pickle.load(open(args.data + "/path_graph.pickle",'rb'))
    #print(kg[700])
    #print(kg[701])
    #print(cb_kg[66])
    # for next_r in cb_kg[66]:
    #     mid_e = torch.Tensor(cb_kg[66][next_r]).long()
    #     #print(mid_e.shape)
    #     mid_e_emd = cb_r[mid_e,:]
    #     #print(mid_e_emd.shape)
    #     mean_mid_e_emd = mid_e_emd.mean(dim = 0)
    #     #print(mean_mid_e_emd.shape)

    #     output = cos(r[66]+mean_mid_e_emd+r[next_r],r[48])
    #     angs = (torch.acos(output)*180/3.1415926).item()
    #     print("r,angs",next_r,angs)

    #test kg
    # print("test kg,for kg[1] and cb_kg[1]...")
    # for tail in kg[1]:
    #     print("1",kg[1][tail],tail)
    # for tail in cb_kg[1]:
    #     print("1",cb_kg[1][tail],tail)

    #返回谓词r和邻接谓词的距离
    # cb_r_emd:cb_r的表征；cb_kg：cb_r的链接拓扑graph
    def mean_dist(cb_r_emd,cb_kg,r_num = 237):
        print("Generating r_dist  >>>")
        dist_r = {}
        for rh in range(r_num):
            if rh not in cb_kg:
                print(rh,"is not a head predicate.")
                dist_r[rh] = None
                continue
            if len(cb_kg[rh])==0:
                dist_r[rh] = None
                continue
            dist_r[rh] = {}
            for next_r in range(r_num):
                if next_r in cb_kg[rh].keys():
                    mid_e = torch.Tensor(cb_kg[rh][next_r]).long()#rh和next_r之间所有cb_r（实体）的id
                    #print(mid_e.shape)
                    mid_e_emd = cb_r_emd[mid_e,:]
                    #print(mid_e_emd.shape)
                    mean_mid_e_emd = mid_e_emd.mean(dim = 0)
                    dist_r[rh][next_r] = mean_mid_e_emd
                else:
                    dist_r[rh][next_r] = None
        return dist_r

    file = args.data + "/r_dist.pickle"
    if not os.path.exists(file):
        r_dist = mean_dist(cb_r,cb_kg)
        with open(file, 'wb') as handle:
            pickle.dump(r_dist, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated r_dist  >>>")
        r_dist = pickle.load(open(file,'rb'))
    #print(r_dist[66])


    def mine_rule(target_r,r_related,r_emd,r_dist,length = 2,r_num = 237):        
        print("mining rules (lenth =",lenth,") for predicate",target_r)
        head_related = r_related[0]
        tail_related = r_related[1]
        context_related = r_related[2]
        minangs = 1000
        rule=[]
        if length == 2:
            rt1= None
            rt2 = None
        elif length ==3:
            rt1 = None
            rt2 = None
            rt3 = None
        if length == 2:
            for rh in head_related[target_r]:
                if r_dist[rh]==None:continue
                if rh == target_r:continue
                for next_r in r_dist[rh].keys():
                    if next_r == target_r:continue
                    if next_r == rh:continue
                    if next_r not in context_related[rh]:continue
                    if next_r not in tail_related[target_r]:continue #长为2的规则适用
                    dist = r_dist[rh][next_r]
                    if dist ==None:continue
                    output = cos(r_emd[rh]+dist+r_emd[next_r],r[target_r])
                    angs = (torch.acos(output)*180/3.1415926).item()
                    #print("rh,next_r,>>,angs",rh,next_r,angs)
                    if angs<minangs:
                        minangs = angs
                        rt1 = rh
                        rt2 = next_r
            rule = [rt1,rt2]        
        elif length == 3:
            for rh in head_related[target_r]:
                if rh == target_r:continue
                if r_dist[rh]==None:continue
                for next_r in r_dist[rh].keys():
                    if next_r == target_r:continue
                    if next_r == rh:continue
                    if next_r not in context_related[rh]:continue
                    #2
                    if next_r not in tail_related[target_r]:#只可能是3规则                        
                        dist1 = r_dist[rh][next_r]
                        if dist1 ==None:continue
                        if r_dist[next_r] ==None:continue
                        
                        for next2_r in r_dist[next_r].keys():                            
                            if next2_r == target_r:continue
                            if next2_r == rh:continue
                            if next2_r == next_r:continue
                            if next2_r not in context_related[next_r]:continue
                            if next2_r not in tail_related[target_r]:continue#仅适用于3规则
                            
                            dist2 = r_dist[next_r][next2_r]
                            if dist2 ==None:continue
                            output = cos(r_emd[rh]+dist1+r_emd[next_r]+dist2+r_emd[next2_r],r[target_r])
                            angs = (torch.acos(output)*180/3.1415926).item()
                            # #print("rh,next_r,>>,angs",rh,next_r,angs)
                            # if target_r == 48 and rh == 66 and (next_r ==135 or next_r == 133):
                            #     print ("3, next_r,next2_r",next_r,next2_r,angs)
                            #     print ("rt1,rt2,rt3",rt1,rt2,rt3)
                            if angs<minangs:
                                minangs = angs
                                rt1 = rh
                                rt2 = next_r
                                rt3 = next2_r
                    else:#可能是2、3规则
                        dist1 = r_dist[rh][next_r]
                        if dist1 ==None:continue
                        output = cos(r_emd[rh]+dist1+r_emd[next_r],r[target_r])
                        angs = (torch.acos(output)*180/3.1415926).item()
                        #print("rh,next_r,>>,angs",rh,next_r,angs)
                        if angs<minangs:
                            minangs = angs
                            rt1 = rh
                            rt2 = next_r
                            rt3 = None
                        if r_dist[next_r] ==None:continue
                        for next2_r in r_dist[next_r].keys():                            
                            if next2_r == target_r:continue
                            if next2_r == rh:continue
                            if next2_r == next_r:continue
                            if next2_r not in context_related[next_r]:continue
                            if next2_r not in tail_related[target_r]:continue#仅适用于3规则
                            
                            dist2 = r_dist[next_r][next2_r]
                            if dist2 ==None:continue
                            output = cos(r_emd[rh]+dist1+r_emd[next_r]+dist2+r_emd[next2_r],r[target_r])
                            angs = (torch.acos(output)*180/3.1415926).item()
                            # #print("rh,next_r,>>,angs",rh,next_r,angs)
                            # if target_r == 48 and rh == 66 and (next_r ==135 or next_r == 133):
                            #     print ("2/3 next_r,next2_r",next_r,next2_r,angs)
                            #     print ("rt1,rt2,rt3",rt1,rt2,rt3)
                            if angs<minangs:
                                minangs = angs
                                rt1 = rh
                                rt2 = next_r
                                rt3 = next2_r
            if rt3 != None:rule = [rt1,rt2,rt3]
            else:rule = [rt1,rt2]     
        return rule,minangs
    #target_r = 48
    lenth = 3
    r_num = 237

    def gen_r_head_tail(kg):
        r_head_e = {}
        r_tail_e = {}
        for e1 in kg.keys():
            for e2 in kg[e1].keys():
                rr_l = kg[e1][e2]
                for rr in rr_l:
                    if rr not in r_head_e:r_head_e[rr]=[]
                    if rr not in r_tail_e:r_tail_e[rr]=[]
                    if e1 not in r_head_e[rr]:r_head_e[rr].append(e1)
                    if e2 not in r_tail_e[rr]:r_tail_e[rr].append(e2)
        return r_head_e,r_tail_e
    
    file = args.data + "/r_head_tail.pickle"
    if not os.path.exists(file):
        r_head_tail = gen_r_head_tail(kg)
        with open(file, 'wb') as handle:
            pickle.dump(r_head_tail, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated r_head_tail  >>>")
        r_head_tail = pickle.load(open(file,'rb'))


    #判断关联的r
    def xx_gen_r_related(kg,r_num = 237):
        #首关联
        head_related = {}
        for r1 in range(r_num):   
            print("Gen head related r for ",r1)
            head_related[r1] = []
            for r2 in range(r_num):
                for e in kg.keys():
                    if r2 in kg[e].items() and r1 in kg[e].items():
                        head_related[r1].append(r2)
                        break
                #if r2 in head_related[r1]:break
            print(r1," head_related = ",head_related[r1])
                #if r2 in head_related[r1]:break
        return head_related
    
    def gen_r_related(r_head_tail,r_num = 237):
        r_head_e = r_head_tail[0]
        r_tail_e = r_head_tail[1]
        #首关联
        head_related = {}
        tail_related = {}
        context_related = {}
        for r1 in range(r_num):   
            print("Gen head/tail/context related r for ",r1)
            head_related[r1] = []
            tail_related[r1] = []
            context_related[r1] = []
            for r2 in range(r_num):
                if r2 == r1:continue
                for e in r_head_e[r1]:
                    if e in r_head_e[r2] and r2 not in head_related[r1]:
                        head_related[r1].append(r2)
                        break
                for e in r_tail_e[r1]:
                    if e in r_tail_e[r2] and r2 not in tail_related[r1]:
                        tail_related[r1].append(r2)
                        break
                for e in r_tail_e[r1]:
                    if e in r_head_e[r2] and r2 not in context_related[r1]:
                        context_related[r1].append(r2)
                        break
        return head_related,tail_related,context_related              

    file = args.data + "/r_related_context.pickle"
    if not os.path.exists(file):
        r_related = gen_r_related(r_head_tail)
        with open(file, 'wb') as handle:
            pickle.dump(r_related, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated r_related_context  >>>")
        r_related = pickle.load(open(file,'rb'))

    print("mining rules,lenth =",lenth)
    Rules = {}
    for target_r in range(r_num):
        bestrule,confidence = mine_rule(target_r,r_related,r,r_dist,lenth,r_num)
        print(target_r,"<-",bestrule,"confidence = ",30/confidence)
        Rules[target_r]=(bestrule,confidence)

    #155 <- [109, 3, 84] confidence =  0.8790513070549716
    # print("head for 155",r_related[0][155])
    # print("context for 109",r_related[2][109])
    # print("context for 3",r_related[2][3])
    # print("tail for 155",r_related[1][155])
    return Rules

def find_path(args):
    model_gat,cb_model_gat = read_cd_gat(args)
    # final_entity_embeddings = model_gat.final_entity_embeddings
    # final_relation_embeddings = model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
    # cb_final_entity_embeddings = cb_model_gat.final_entity_embeddings
    # cb_final_relation_embeddings = cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
    entity_embeddings = model_gat.entity_embeddings
    relation_embeddings = model_gat.relation_embeddings
    e_ijk_emd = model_gat.final_out_entity_l_1#entity embedding在ijk上的分量值[200, 272115]
    cb_entity_embeddings = cb_model_gat.entity_embeddings
    cb_relation_embeddings = cb_model_gat.relation_embeddings
    r_uve_emd = cb_model_gat.final_out_entity_l_1#relation embedding在uve上的分量值[200, 272115]
    
    cubic_attention = model_gat.cubic_attention
    cb_cubic_attention = cb_model_gat.cubic_attention

    adj = Corpus_.train_adj_matrix
    cb_adj = cb_Corpus_.train_adj_matrix
    edge_list = adj[0] #e2,e1
    edge_type = adj[1] #r
    cb_edge_list = cb_adj[0] #r2,r1
    cb_edge_type = cb_adj[1] #cb_r
    edge_embed = relation_embeddings[edge_type]#1hop的关系嵌入

    N = cb_entity_embeddings.size()[0] #237
    out_features = 1
    a = torch.sparse_coo_tensor(
            cb_edge_list, cb_cubic_attention, torch.Size([N, N, out_features]))


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

def TestRules_restrict(Rules):
    #test_indices = np.array(list(Corpus_.test_triples)).astype(np.int32)
    kg = Corpus_.get_multiroute_graph()
    
    num = len(Corpus_.test_indices)#所有的三元组
    
    grnd = 0#所有的ground连接数
    #grnd_all = 0#所有ground连接数
    #grnd_res = 0
    num_r = {}
    grnd_r = {}#谓词r的ground连接数（字典）
    #grnd_r_all = {}#所有ground连接数
    real_r = {}#根据不严格判断，hit的实体数
    real_r_res = {}#根据严格判断，hit的实体数
    for iters in range(1):
        start_time = time.time()
        indices = [i for i in range(num)]
        batch_indices = Corpus_.test_indices[indices, :]
        print("Sampled indices")
        print("test set length ", num)
        entity_list = [j for i, j in Corpus_.entity2id.items()]
        for i in range(batch_indices.shape[0]):
            h = batch_indices[i, 0]
            t = batch_indices[i, 2]
            r = batch_indices[i, 1]
            if r not in grnd_r.keys():
                grnd_r[r] = 0
                #grnd_r_all[r] = 0
            if r not in num_r.keys():
                num_r[r] = 0
            if r not in real_r.keys():
                real_r[r] = 0
                real_r_res[r] = 0
            num_r[r] += 1
            rule = Rules[r][0]
            rule_confd = Rules[r][1]
            #for rx in rule:
            if len(rule)==2:
                flag0_res = 1 
                flag0 = 0
                if h not in kg.keys():
                    flag0_res = 0 
                    continue  
                for e2 in kg[h].keys():
                    flag1 = 0
                    if rule[0] in kg[h][e2]: 
                        if e2 not in kg.keys():
                            #flag1 = 0 
                            continue
                        for e3 in kg[e2].keys():
                            if rule[1] in kg[e2][e3]:
                                if e3 == t:
                                    grnd += 1
                                    grnd_r[r] += 1
                                    flag1 = 1 
                                    break
                    if flag1==0:
                        flag0_res = 0
                        
                    if flag1==1:
                        flag0 = 1
                        break
                if flag0>0:
                    real_r[r] += 1
                if flag0_res>0:
                    real_r_res[r] += 1                

            if len(rule)==3:
                flag0_res = 1 
                flag0 = 0                
                if h not in kg.keys():
                    flag0_res = 0 
                    continue        
                for e2 in kg[h].keys():
                    flag1_res = 1 
                    flag1 = 0
                    if rule[0] in kg[h][e2]:                        
                        if e2 not in kg.keys():
                            #flag1_res = 0 
                            continue
                        for e3 in kg[e2].keys():                            
                            flag2 = 0
                            if rule[1] in kg[e2][e3]:                                                      
                                if e3 not in kg.keys():
                                    #flag2 = 0 
                                    continue
                                for e4 in kg[e3].keys():
                                    if rule[2] in kg[e3][e4]:
                                        if e4 == t:
                                            grnd += 1
                                            grnd_r[r] += 1
                                            flag2 = 1 
                                            break
                            if flag2==0:
                                flag1_res = 0
                                
                            if flag2==1:
                                flag1 = 1
                                break
                    if flag1_res==0:
                        flag0_res = 0
                        
                    if flag1==1:
                        flag0 = 1
                        break
                if flag0>0:
                    real_r[r] += 1
                if flag0_res>0:
                    real_r_res[r] += 1
    count = 0
    hit = 0
    hit_restrict = 0
    grnd_cnt = 0  
    for r in Rules.keys():
        
        print("r:",r,"rules:", Rules[r])
        if r not in num_r:
            print("No related tripples.")
            continue
        print("Number of related tripples:",num_r[r])
        print("Number of grouding truth:",grnd_r[r],"Number of Hits:",real_r[r],"Number of restricted Hits:", real_r_res[r])
        print("confidence:",real_r[r]/num_r[r],"restricted confidence:",real_r_res[r]/num_r[r])
        if grnd_r[r]>0:
            count += num_r[r]
            grnd_cnt += grnd_r[r]
            hit += real_r[r]
            hit_restrict += real_r_res[r]
    print("Total Confidence:",hit/count,"restricted confidence:",hit_restrict/count,"grouding truth:",grnd_cnt)
                    
if __name__ == "__main__":
    #ShortestPath(args)
    #find_path(args)

    file = args.data + "/Rules0330.pickle"
    if not os.path.exists(file):
        Rules = ShortestPath(args)
        with open(file, 'wb') as handle:
            pickle.dump(Rules, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated Rules  >>>")
        Rules = pickle.load(open(file,'rb'))
    TestRules_restrict(Rules)