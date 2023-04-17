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
                           for _ in range(nheads)] #多head的attention，使用本step更新前的关系嵌入

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        #[100,200]or[50,200] 固定把模型的relation的embedding转为200维度的输出embedding
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid))) #关系嵌入的学习参数矩阵WR
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             ) #完成多head到1head的聚合，其中采用本step更新后的关系嵌入
    # edge_list, edge_type, edge_embed为1hop的e2(t)、e1(h)，关系r，以及关系r的嵌入
    #edge_list_nhop, edge_type_nhop为2hop的首尾实体e1、e2_hop、关系路径r1_2hop,r2_2hop
    #entity_embeddings, relation_embed为嵌入表
    #
    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        if(edge_type_nhop.shape[0]>0):
            edge_embed_nhop = relation_embed[
                edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]] #对nhop路径直接相加，获得2hop关系嵌入（类transE）
        else:
            edge_embed_nhop = torch.tensor([]).float()
        #print("x.shape bf multi-att",x.shape)
        #edge_list = e2,e1 = t,h
        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)[0]
                       for att in self.attentions], dim=1) #对多head的attention，直接concat
        print("x.shape aft multi-att",x.shape)
        x = self.dropout_layer(x)#x中每个head是更新后的节点隐变量hi。

        out_relation_1 = relation_embed.mm(self.W)#关系的嵌入学习，只是乘以一个WR.[237,200],[14451,200]

        edge_embed = out_relation_1[edge_type]
        if(edge_type_nhop.shape[0]>0):
            edge_embed_nhop = out_relation_1[
                edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]] #关系的嵌入学习，只是乘以一个WR
        x,x_l,x_w = self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop)#x:[14541, 50];edge_embed[272115, 50]
        x = F.elu(x) #将多head的attention转为输出维度的嵌入，其中采用本step更新后的关系嵌入.[14541, 200] or [237, 200]
        x_l = F.elu(x_l) #[200, 272115]
        x_w = F.elu(x_w) #[200, 272115]
        return x, out_relation_1,x_l,x_w #输出训练后的实体嵌入表、关系嵌入表。


class SpKBGATModified(nn.Module):#initial_entity_emb、initial_relation_emb决定了模型运行时实体、关系的embedding
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

        self.num_nodes = initial_entity_emb.shape[0] #实体量14541 =  [14541, 100][0] 或237=  [237, 100][0] (pretrain时固定为100，其他时候随args)
        self.entity_in_dim = initial_entity_emb.shape[1]    # 100 =  [14541, 100][1]    或100=  [237, 100][1]
        self.entity_out_dim_1 = entity_out_dim[0] #输出实体的维度1为100， =[100,200][0]
        self.nheads_GAT_1 = nheads_GAT[0] # =[2,2]
        self.entity_out_dim_2 = entity_out_dim[1]#输出实体的维度2为200 =[100,200][1]
        self.nheads_GAT_2 = nheads_GAT[1]
        #print("SpKBGATModified__init__ initial_entity_emb.shape,initial_relation_emb.shape",initial_entity_emb.shape,initial_relation_emb.shape) #[14541, 50]/[237, 200]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]#关系量237 =  [237, 100][0]  或14541=  [14541, 50][0]
        self.relation_dim = initial_relation_emb.shape[1]# 100 =  [237, 100][1]  或50=  [14541, 50][1]
        self.relation_out_dim_1 = relation_out_dim[0]#输出关系的维度1 =100，未用？

        self.drop_GAT = drop_GAT #drop率
        self.alpha = alpha      # For leaky relu
        
        #[14541,200]或[237,200]
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))#实体最终嵌入的初始值，放入final_entity_embeddings
        
        #[237,200]或[14541,200]
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))#关系最终嵌入的初始值

        if cb_flag:
            self.entity_embeddings = initial_entity_emb #[237, 100](pretrain时固定为100，其他时候随args默认50)
        else:
            self.entity_embeddings = nn.Parameter(initial_entity_emb)#实体嵌入（过程量）的初始值 #[14541, 100](pretrain时固定为100，其他时候随args默认50)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)#[237, 100]或 [14541, 50]#前者pretrain时固定为100，其他时候随args默认50；后者一直随args默认50

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        #[100,200](pretrain时固定为100，其他时候随args默认50)
        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1))) #参数WE，用于对最终嵌入加成原始特征
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        #[100,200*237]或[100,200*14541](pretrain时固定为100，其他时候随args默认50)
        self.W_entities_l = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1*self.num_relation))) #参数WE，用于对最终嵌入加成原始特征
        nn.init.xavier_uniform_(self.W_entities_l.data, gain=1.414)
    #adj形式为(adj_indices, adj_values)，每行是([e1,e2],r),r为关系id或者1
    #batch_inputs为转为int格式的训练三元组id数据
    #train_indices_nhop为当前batch实体的2hop邻居实体，list形式，其中元素为一个2hop路径e1,relations[-1],relations[0],e2
    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop,ass_ent=None,ass_rel=None,ass_rel_l =None):
        #if ass_ent !=None:self.entity_embeddings=nn.Parameter(ass_ent)
        #if ass_rel !=None:self.relation_embeddings=nn.Parameter(ass_rel)
        # getting edge list,adj = ((e2,e1),r) = ((t,h),r)
        edge_list = adj[0] #e2,e1 = t,h
        edge_type = adj[1] #r
        #print("SpKBGATModified__forward__train_indices_nhop.shape",train_indices_nhop.shape)
        if(train_indices_nhop.shape[0]>0):
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()#2hop的首尾实体
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)#2hop的关系路径
        else:
            edge_list_nhop = torch.tensor([]).long()
            edge_type_nhop = torch.tensor([]).long()
        if(CUDA):
            edge_list = edge_list.cuda()#1hop的e2,e1 = t,h
            edge_type = edge_type.cuda()#1hop的r
            edge_list_nhop = edge_list_nhop.cuda()#2hop的首尾实体
            edge_type_nhop = edge_type_nhop.cuda()#2hop的关系路径

        edge_embed = self.relation_embeddings[edge_type]#1hop的关系嵌入

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)
        #
        out_entity_1, out_relation_1,out_entity_l_1,out_w = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop) # GAT的具体过程，见spGAT类

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0
        mask_l = torch.ones(self.relation_embeddings.shape[0]).cuda()
        #print("SpKBGATModified__forward__ mask.shape",mask.shape,mask_l.shape)
        #无pretrain：entity_embeddings.shape = [14541, 50] or [237, 50] relation_embeddings.shape = [237, 50] or [14541, 50]
        #pretrain：entity_embeddings.shape = [14541, 100] or [237, 100] relation_embeddings.shape = [237, 100] or [14541, 50]
        # print("SpKBGATModified__forward__self.entity_embeddings.shape,self.relation_embeddings.shape",self.entity_embeddings.shape,self.relation_embeddings.shape)
        # print("SpKBGATModified__forward__out_entity_1.shape",out_entity_1.shape)#[14541, 200] or [237, 200]
        # print("SpKBGATModified__forward__out_entity_l_1.shape",out_entity_l_1.shape)#[200, 272115] or [200, 502958]
        entities_upgraded = self.entity_embeddings.mm(self.W_entities)#pretrain时：[14541, 100]*[100,200] or [237, 100]*[100,200]
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1 #固定为[14541,200]or [237, 200]其中，200为100*2，2头
        #entities_l_upgraded = self.entity_embeddings.mm(self.W_entities_l)
        #out_entity_l_1 = entities_l_upgraded + \
        #    mask_l.unsqueeze(-1).expand_as(out_entity_l_1) * out_entity_l_1 #
        #out_entity_l_1 = entities_l_upgraded + out_entity_l_1
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)  # 返回值entity_embed, relation_embed，用于transE的score
        out_entity_l_1 = F.normalize(out_entity_l_1, p=2, dim=1) #embedding在ijk上的分量值
        #out_w = F.normalize(out_w, p=2, dim=1)#ijk的attention
        self.final_entity_embeddings.data = out_entity_1.data#固定为[14541,200]or [237, 200]其中，200为100*2，2头
        self.final_relation_embeddings.data = out_relation_1.data #将输出的嵌入的值赋值给实体最终嵌入，注，该tensor是脱离GAT部分的backprop循环的
        self.final_out_entity_l_1 = out_entity_l_1.data
        self.cubic_attention = out_w.data
        return out_entity_1, out_relation_1,out_entity_l_1

#基于ConvKB封装
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
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
