import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,#为100， =[100,200][0]
                                                 relation_dim,#initial_relation_emb.shape[1]# 100 =  [237, 100][1]  或50=  [14541, 50][1]
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)] #

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        #[100,200]or[50,200] 固定把模型的relation的embedding转为200维度的输出embedding
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid))) #WR
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             ) 
    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        if(edge_type_nhop.shape[0]>0):
            edge_embed_nhop = relation_embed[
                edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        else:
            edge_embed_nhop = torch.tensor([]).float()

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)[0]
                       for att in self.attentions], dim=1)
        print("x.shape aft multi-att",x.shape)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)#[237,200],[14451,200]

        edge_embed = out_relation_1[edge_type]
        if(edge_type_nhop.shape[0]>0):
            edge_embed_nhop = out_relation_1[
                edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]] 
        x,x_l,x_w = self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop)#x:[14541, 50];edge_embed[272115, 50]
        x = F.elu(x) #[14541, 200] or [237, 200]
        x_l = F.elu(x_l) #[200, 272115]
        x_w = F.elu(x_w) #[200, 272115]
        return x, out_relation_1,x_l,x_w 


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT,cb_flag = False):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0] #entity 14541 =  [14541, 100][0] 或237=  [237, 100][0]
        self.entity_in_dim = initial_entity_emb.shape[1]    # 100 =  [14541, 100][1]    或100=  [237, 100][1]
        self.entity_out_dim_1 = entity_out_dim[0] # =[100,200][0]
        self.nheads_GAT_1 = nheads_GAT[0] # =[2,2]
        self.entity_out_dim_2 = entity_out_dim[1]# =[100,200][1]
        self.nheads_GAT_2 = nheads_GAT[1]
        #print("SpKBGATModified__init__ initial_entity_emb.shape,initial_relation_emb.shape",initial_entity_emb.shape,initial_relation_emb.shape) #[14541, 50]/[237, 200]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]#rel 237 =  [237, 100][0]  或14541=  [14541, 50][0]
        self.relation_dim = initial_relation_emb.shape[1]# 100 =  [237, 100][1]  或50=  [14541, 50][1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu
        
        #[14541,200]或[237,200]
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))
        
        #[237,200]或[14541,200]
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        if cb_flag:
            self.entity_embeddings = initial_entity_emb #[237, 100]
        else:
            self.entity_embeddings = nn.Parameter(initial_entity_emb)#[14541, 100]
        self.relation_embeddings = nn.Parameter(initial_relation_emb)#[237, 100]或 [14541, 50]

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        #[100,200]
        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1))) #WE
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        #[100,200*237]或[100,200*14541]
        self.W_entities_l = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1*self.num_relation))) #WE
        nn.init.xavier_uniform_(self.W_entities_l.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop,ass_ent=None,ass_rel=None,ass_rel_l =None):

        edge_list = adj[0] #e2,e1 = t,h
        edge_type = adj[1] #r

        if(train_indices_nhop.shape[0]>0):
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        else:
            edge_list_nhop = torch.tensor([]).long()
            edge_type_nhop = torch.tensor([]).long()
        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()



        out_entity_1, out_relation_1,out_entity_l_1,out_w = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0
        mask_l = torch.ones(self.relation_embeddings.shape[0]).cuda()

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)#[14541, 100]*[100,200] or [237, 100]*[100,200]
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1 #[14541,200]or [237, 200]

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1) 
        out_entity_l_1 = F.normalize(out_entity_l_1, p=2, dim=1)
        self.final_entity_embeddings.data = out_entity_1.data#[14541,200]or [237, 200]
        self.final_relation_embeddings.data = out_relation_1.data
        self.final_out_entity_l_1 = out_entity_l_1.data
        self.cubic_attention = out_w.data
        return out_entity_1, out_relation_1,out_entity_l_1

class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1) #shape = batch_size, length, dim
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
