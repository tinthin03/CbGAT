import torch
import os
import numpy as np


def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    #逆关系
    R = len(relation2id.keys())
    inv_relation2id = dict()
    for relation,relation_id in relation2id.items():
        inv_relation2id["inv_"+relation] = int(relation_id)+R
    relation2id.update(inv_relation2id)
    # R = len(relation2id.keys())
    # print(R) 474

    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])
    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val)*-1 for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

#载入数据文件，形成三元组
#triples_data每行是一个三元组的id，(rows, cols, data)分别是目标实体、源实体、关系的id(或者1)，当是1时，可以组合成一个邻接矩阵
#对于(h,r,t),邻接矩阵(rows, cols, data)的每行，rows:t,cols:h,data:r
def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        triples_data.append(
            (entity2id[e2], relation2id["inv_"+relation], entity2id[e1]))

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id["inv_"+relation])

        

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)

#生成训练所需的三元组和邻接矩阵
def build_data(path='./data/WN18RR/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    #返回：三元组id(head,tail,r)， (rows, cols, data)=(tail,head,r)，出现过的实体名
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}#实体序号逆字典
    id2relation = {v: k for k, v in relation2id.items()}#
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()
    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1 #(e1, relation)左连接的数量

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1 #(relation, e2)右连接的数量

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i]) #每个关系在左连接中的平均数（关系i连接每种源实体的平均数量）

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i]) #每个关系在右连接中的平均数（关系i连接每种目标实体的平均数量）

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i]) #表示关系尾部链接实体的平均数量与头部链接实体的平均数量之比。

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train

#生成训练所需的三元组和邻接矩阵
#对于(h,r,t),邻接矩阵train_cb_adjacency_mat的每行，cb_rows:t,cb_cols:h,cb_data:r
def build_cubic_data(path='./data/WN18RR/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')
    R = int(len(relation2id.keys())/2) # 237
    E = int(len(entity2id.keys()))
    cb_entity2id = relation2id
    cb_relation2id = entity2id

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    #返回：三元组id， (rows, cols, data)，出现过的实体名
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}#实体序号逆字典
    id2relation = {v: k for k, v in relation2id.items()}#
    left_entity, right_entity = {}, {}
    id2cd_entity = {v: k for k, v in relation2id.items()}
    id2cd_relation = {v: k for k, v in entity2id.items()}
    left_cb_entity, right_cb_entity = {}, {}
    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()
    cubic_rel = {}
    train_cb_triples = []
    train_cb_adjacency_mat = ()
    cb_rows, cb_cols, cb_data = [], [], []
    unique_cb_entities = set()
    print("Num of triples",len(lines))
    count = len(lines)
    indx = 0
    for line in train_triples:
        e1_id, rel_id, e2_id = line#读出为实体、关系的名字，需转为数字id
        e1 = id2entity[e1_id]
        relation = id2relation[rel_id]
        e2 = id2entity[e2_id]
        if indx % 1000== 0:
            print("Gen triple:e1, relation, e2 :",indx,'/',count,'   ',e1_id,e1,'   ',rel_id, relation,'   ', e2_id,e2)
        for tri in train_triples:#扫描训练集，抽取cubic关系
            #print("tri:",tri)
            if e1_id == tri[2]:
                if ((tri[1], e1_id, rel_id)) not in train_cb_triples:
                    train_cb_triples.append((tri[1], e1_id, rel_id))
                    unique_cb_entities.add(id2cd_entity[tri[1]])
                    unique_cb_entities.add(id2cd_entity[rel_id])

                    #Connecting tail and source entity
                    cb_rows.append(rel_id)
                    cb_cols.append(tri[1])
                    if is_unweigted:
                        cb_data.append(1)
                    else:
                        cb_data.append(e1_id)                   

                    if indx % 100== 0:
                        print("     Gen cubic ent ():",tri[1], e1_id, rel_id)
            
            if e2_id == tri[0]:
                if ((rel_id, e2_id, tri[1])) not in train_cb_triples:
                    train_cb_triples.append((rel_id, e2_id, tri[1]))
                    unique_cb_entities.add(id2cd_entity[tri[1]])
                    unique_cb_entities.add(id2cd_entity[rel_id])

                    # Connecting tail and source entity
                    cb_rows.append(tri[1])
                    cb_cols.append(rel_id)
                    if is_unweigted:
                        cb_data.append(1)
                    else:
                        cb_data.append(e2_id)
                    

                    if indx % 100== 0:
                        print("     Gen cubic ent:",rel_id, e2_id, tri[1])
            
        indx += 1
        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1 #(e1, relation)左连接的数量

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1 #(relation, e2)右连接的数量

    train_cb_adjacency_mat = (cb_rows, cb_cols, cb_data)

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i]) #每个关系在左连接中的平均数（关系i连接每种源实体的平均数量）

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i]) #每个关系在右连接中的平均数（关系i连接每种目标实体的平均数量）

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i]) #表示关系尾部链接实体的平均数量与头部链接实体的平均数量之比。
    cb_headTailSelector = {}
    for i in range(len(cb_relation2id)):
        cb_headTailSelector[i] = 1000 * 0.5 #表示关系尾部链接实体的平均数量与头部链接实体的平均数量之比。

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train,(train_cb_triples, train_cb_adjacency_mat),cb_entity2id,cb_relation2id,cb_headTailSelector,list(unique_cb_entities)