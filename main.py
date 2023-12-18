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



model_type = 'rotatE'
inductive = False
# exp = "fb15"
# exp = "wn18"
# exp = "umls"
# exp = "ilpc"
exp = "ilpc-large"

inv_relatation = False #data & reverse data

reverse = False # reverse data


if reverse:suf = '-reverse'
elif inv_relatation :suf = '-inv'
else: suf = ''

if exp == "ilpc" or exp == "ilpc-large":
    inductive = True

def parse_args_wn18():
    args = argparse.ArgumentParser()
    # network arguments
    
    if model_type=='rotatE':
        args.add_argument("-data", "--data",
                        default="./data/WN18RR-rotate/", help="data directory")
        
        args.add_argument("-outfolder", "--output_folder",
                        default="./checkpoints/wn/out-rotate/", help="Folder name to save the models.")
    
    else:
        args.add_argument("-data", "--data",
                        default="./data/WN18RR/", help="data directory")
        
        args.add_argument("-outfolder", "--output_folder",
                        default="./checkpoints/wn/out/", help="Folder name to save the models.")
    
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)

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
    if model_type=='rotatE':
        if inductive:
            args.add_argument("-data", "--data",
                        default=f"./data/FB15k-237-rotate{suf}/inductive/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                    default=f"./checkpoints/fb/out-rotate{suf}-inductive/", help="Folder name to save the models.") #   /out-rotate-inductive/
        else:            
            args.add_argument("-data", "--data",
                        default=f"./data/FB15k-237-rotate{suf}/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                    default=f"./checkpoints/fb/out-rotate{suf}/", help="Folder name to save the models.") #   /out-rotate-inductive/ #XXX inv200  -dim100
        
    else:
        if inductive:
            args.add_argument("-data", "--data",
                            default=f"./data/FB15k-237{suf}/inductive2/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default=f"./checkpoints/fb/out{suf}-inductive2/", help="Folder name to save the models.") #out-inductive/ -rotate
        else:
            args.add_argument("-data", "--data",
                            default=f"./data/FB15k-237{suf}/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default=f"./checkpoints/fb/out{suf}/", help="Folder name to save the models.") #out-inductive/ -rotate
        
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=(not reverse) and (not inv_relatation), help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=10000, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    if inv_relatation:
        args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                        default=[50, 100], help="Entity output embedding dimensions")
    else:
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


def parse_args_umls():
    args = argparse.ArgumentParser()
    
    # network arguments
    if model_type=='rotatE':
        if inductive:
            args.add_argument("-data", "--data",
                        default="./data/umls-rotate/inductive/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/umls/out-rotate-inductive/", help="Folder name to save the models.") #   /out-rotate-inductive/
        else:            
            args.add_argument("-data", "--data",
                        default="./data/umls-rotate/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/umls/out-rotate/", help="Folder name to save the models.") #   /out-rotate-inductive/
    else:
        if inductive:
            args.add_argument("-data", "--data",
                            default="./data/umls/inductive/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/umls/out-inductive/", help="Folder name to save the models.") #out-inductive/ -rotate
        else:
            args.add_argument("-data", "--data",
                            default="./data/umls/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/umls/out/", help="Folder name to save the models.") #out-inductive/ -rotate
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)

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

def parse_args_ilpc():
    args = argparse.ArgumentParser()
    
    # network arguments
    if model_type=='rotatE':
        if inductive:
            args.add_argument("-data", "--data",
                        default="./data/ilpc-small-rotate/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/ilpc/ilpc-small-rotate/", help="Folder name to save the models.") #   /out-rotate-inductive/
        else:            
            args.add_argument("-data", "--data",
                        default="./data/ilpc-small-rotate/transductive/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/ilpc/ilpc-small-rotate-transductive/", help="Folder name to save the models.") #   /out-rotate-inductive/
    else:
        if inductive:
            args.add_argument("-data", "--data",
                            default="./data/ilpc-small/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/ilpc/ilpc-small/", help="Folder name to save the models.") #out-inductive/ -rotate
        else:
            args.add_argument("-data", "--data",
                            default="./data/ilpc-small/transductive/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/ilpc/ilpc-small-transductive/", help="Folder name to save the models.") #out-inductive/ -rotate
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=3000, help="Batch size for GAT")
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

def parse_args_ilpc_large():
    args = argparse.ArgumentParser()
    
    # network arguments
    if model_type=='rotatE':
        if inductive:
            args.add_argument("-data", "--data",
                        default="./data/ilpc-large-rotate/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/ilpc/ilpc-large-rotate/", help="Folder name to save the models.") #   /out-rotate-inductive/
        else:            
            args.add_argument("-data", "--data",
                        default="./data/ilpc-large-rotate/transductive/", help="data directory")  #FB15k-237-rotate/inductive/
            args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/ilpc/ilpc-large-rotate-transductive/", help="Folder name to save the models.") #   /out-rotate-inductive/
    else:
        if inductive:
            args.add_argument("-data", "--data",
                            default="./data/ilpc-large/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/ilpc/ilpc-large/", help="Folder name to save the models.") #out-inductive/ -rotate
        else:
            args.add_argument("-data", "--data",
                            default="./data/ilpc-large/transductive/", help="data directory")#FB15k-237-direct-pretr  inductive/  -rotate
            args.add_argument("-outfolder", "--output_folder",
                            default="./checkpoints/ilpc/ilpc-large-transductive/", help="Folder name to save the models.") #out-inductive/ -rotate
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=5000, help="Batch size for GAT")
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

if exp == 'fb15':
    args = parse_args()
elif exp == 'wn18':
    args = parse_args_wn18()
elif exp == 'umls':
    args = parse_args_umls()
elif exp == 'ilpc':
    args = parse_args_ilpc()
elif exp == 'ilpc-large':
    args = parse_args_ilpc_large()
# %%

cubic_from_reverse = False#True

def load_data(args):
    file = args.data + "/load_data_pickle.pickle"
    if os.path.exists(file):
        with open(file, 'rb') as handle:
            load_data_pickle = pickle.load(handle)
            train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,unique_cb_entities = load_data_pickle
                   
    else:
        train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,\
            unique_cb_entities = build_cubic_data(args.data, is_unweigted=False, directed=(not inv_relatation),cubic_from_reverse=cubic_from_reverse,reverse_load_data = "reverse_load_data_pickle.pickle")

        load_data_pickle = (train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train,\
            train_cb_data,cb_entity2id,cb_relation2id,cb_headTailSelector,unique_cb_entities)

        file = args.data + "/load_data_pickle.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(load_data_pickle, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'),inv = inv_relatation)
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")
    cb_entity_embeddings = relation_embeddings

    cb_relation_embeddings = np.random.randn(len(entity2id), args.embedding_size)#
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    cb_corpus = Corpus(args, train_cb_data, validation_data, test_data, cb_entity2id, cb_relation2id, cb_headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_cb_entities, args.get_2hop,True)
    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings),cb_corpus,torch.FloatTensor(cb_entity_embeddings), torch.FloatTensor(cb_relation_embeddings)

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

if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)


entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)
cb_entity_embeddings_copied = deepcopy(cb_entity_embeddings)
cb_relation_embeddings_copied = deepcopy(cb_relation_embeddings)
print("Initial cb_entity dimensions {} , cb_relation dimensions {}".format(
    cb_entity_embeddings.size(), cb_relation_embeddings.size()))
print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

cos = torch.nn.CosineSimilarity(dim=0,eps=1e-12)
def infer(h,r,model = 'transE'):
    if model=='transE':
        return h+r
    else:
        re_h, im_h = torch.chunk(h, 2, dim=-1)

        pi = 3.141592653589793238462643383279
        r = r / (1 / pi)#

        r_up,r_dn = torch.chunk(r, 2, dim=-1)

        r_conj = torch.min(r_up-r_dn,r_up+r_dn)
        re_r = torch.cos(r_conj)
        im_r = torch.sin(r_conj)

        re_res = re_h * re_r - im_h * im_r
        im_res = re_h * im_r + im_h * re_r

        return torch.cat([re_res, im_res], dim=-1)
def chain(r1,r2,model = 'transE'):
    return r1+r2
def dist(c,t,model = 'transE',sim_model = 'F1'):
    if model=='transE':
        if sim_model == 'cos':
            dist = cos(c,t)            
        elif sim_model == 'F2':
            dist = torch.linalg.norm((c)-t,ord=2, dim=1) #ord=2
        else:#F1
            dist = torch.linalg.norm((c)-t,ord=1, dim=1) #ord=1
    else:
        a = c - t
        re, im = torch.chunk(a, 2, dim=-1)
        a = torch.stack([re, im], dim=-1)
        dist = a.norm(dim=-1).sum(dim=-1)
        
    return dist#shape=c.shape(0)=t.shape(0)

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed,mod = 0):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))
    print("train_indices.shape",train_indices.shape)
    print("len_pos_triples",len_pos_triples)

    pos_triples = train_indices[:len_pos_triples]
    if mod == 0:
        neg_triples = train_indices[len_pos_triples:]
    else:
        neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    pos_norm = dist(infer(source_embeds,relation_embeds,model = model_type),tail_embeds,model = model_type)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    neg_norm = dist(infer(source_embeds,relation_embeds,model = model_type),tail_embeds,model = model_type)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


device_ids=[0,1]

def train_gat(args):
    print("train_gat ...")
    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT
    if CUDA:
        model_gat.cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
    

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([]).long()

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
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
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
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            print("\n len(train_indices) ==", len(train_indices))
            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
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
    #epoch_load = args.epochs_gat
    epoch_load = 0#
    epoch_load = 0#

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT

    pre  = 'cb_e'
    if epoch_load>0:
        model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_e{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    final_entity_embeddings = model_gat.final_entity_embeddings
    final_relation_embeddings = model_gat.final_relation_embeddings#

    
    cb_model_gat = SpKBGATModified(model_gat.relation_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    args.drop_GAT, args.alpha, args.nheads_GAT,cb_flag=True) #cubic GAT

    pre  = 'cb_r'
    if epoch_load>0:
        cb_model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_r{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    cb_final_entity_embeddings = cb_model_gat.final_entity_embeddings
    cb_final_relation_embeddings = cb_model_gat.final_relation_embeddings#

    divc = torch.cuda.current_device()
    if CUDA:
        model_gat.cuda()
        cb_model_gat.cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
            cb_model_gat = nn.DataParallel(cb_model_gat, device_ids=[torch.cuda.current_device()])
        model_gat  = model_gat.to(divc)
        cb_model_gat  = cb_model_gat.to(divc)
    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    cb_optimizer = torch.optim.Adam(
        cb_model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)


    indiv_params =  []
    idx = 0
    for p in cb_model_gat.parameters():
        if idx!=2:indiv_params += [p]
        idx += 1     
    all_optimizer = torch.optim.Adam([{'params': model_gat.parameters()},
                             {'params': indiv_params}],
                             lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    cb_scheduler = torch.optim.lr_scheduler.StepLR(
        cb_optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    all_scheduler = torch.optim.lr_scheduler.StepLR(
        all_optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([]).long()

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
    cb_entity_embed = cb_entity_embeddings #

    for epoch in range(epoch_load,args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        random.shuffle(cb_Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32) 
        cb_Corpus_.train_indices = np.array(
            list(cb_Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        cb_model_gat.train()
        start_time = time.time()
        epoch_loss = []
        if len(Corpus_.train_indices) < len(cb_Corpus_.train_indices):
            if len(Corpus_.train_indices) % args.batch_size_gat == 0:
                num_iters_per_epoch = len(
                    Corpus_.train_indices) // args.batch_size_gat
            else:
                num_iters_per_epoch = (
                    len(Corpus_.train_indices) // args.batch_size_gat) + 1
        else:
            if len(cb_Corpus_.train_indices) % args.batch_size_gat == 0:
                num_iters_per_epoch = len(
                    cb_Corpus_.train_indices) // args.batch_size_gat
            else:
                num_iters_per_epoch = (
                    len(cb_Corpus_.train_indices) // args.batch_size_gat) + 1
        print("\n num_iters_per_epoch ==", num_iters_per_epoch)


        for iters in range(num_iters_per_epoch):
            print("\n iters-> ", iters)
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)#
            cb_train_indices, cb_train_values = cb_Corpus_.get_iteration_batch(iters) #

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()#.to(divc)
                train_values = Variable(torch.FloatTensor(train_values)).cuda()#.to(divc)
                cb_train_indices = Variable(
                    torch.LongTensor(cb_train_indices)).cuda()#.to(divc)
                cb_train_values = Variable(torch.FloatTensor(cb_train_values)).cuda()#.to(divc)

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))
                cb_train_indices = Variable(torch.LongTensor(cb_train_indices))
                cb_train_values = Variable(torch.FloatTensor(cb_train_values))

            # forward pass
            print("\n Run model_gat forward",)
            if init_flag:
                entity_embed, relation_embed,entity_l_embed = model_gat(
                    Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)
                init_flag = False
            else:
                entity_embed, relation_embed,entity_l_embed = model_gat(
                    Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices,ass_rel=cb_entity_embed)

            print("\n Run cb_model_gat forward",)
            
            cb_entity_embed, cb_relation_embed,cb_entity_l_embed = cb_model_gat(
                cb_Corpus_, cb_Corpus_.train_adj_matrix, cb_train_indices, current_batch_2hop_indices,ass_ent=relation_embed)
            
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


        all_scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch%100==0 or epoch == args.epochs_gat - 1:
            save_model(model_gat, args.data, epoch,
                    args.output_folder,prex = 'cb_e')
            save_model(cb_model_gat, args.data, epoch,
                   args.output_folder,prex = 'cb_r')


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
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv 
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss() 

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []
        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

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

def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities) 


import os

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
    train_gat_cb(args)
    print("train_gat_cb complete")
    inductive = False
    if inductive == False:
        train_conv(args)
        evaluate_conv(args, Corpus_.unique_entities_train)
