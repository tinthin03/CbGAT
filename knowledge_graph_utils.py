import torch

from extra.data.graph import Graph


def load_dataset(DATA_DIR,exp = "fb"):
    entity2id = dict()
    relation2id = dict()

    with open(f'{DATA_DIR}/entity2id.txt') as fin:
        entity2id = dict()
        for line in fin:
            entity,eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(f'{DATA_DIR}/relation2id.txt') as fin:
        relation2id = dict()
        for line in fin:
            #relation,rid = line.strip().split('\t')
            relation,rid = line.strip().split()
            relation2id[relation] = int(rid)

    E = len(entity2id)
    R = len(relation2id)

    mov = R
    R += mov
    # R += 1

    ret = dict()
    ret['E'] = E#实体数
    ret['R'] = R#关系数，包括逆关系
    ret['T'] = dict()#每个关系对应的tripples数量
    ret['Rh']= dict()#每个关系对应的head
    ret['Rt']= dict()#每个关系对应的tail

    for r_id in range(R):
        ret['T'][r_id]=0
        ret['Rh'][r_id]=set()
        ret['Rt'][r_id]=set()
    
    if exp == 'ilpc' or exp == 'ilpc-large':
        Graphs = ['train', 'valid', 'test', 'inference']
    else:
        Graphs = ['train', 'valid', 'test']
    for item in Graphs:
        edges = []
        if item == 'train':
            with open(f"{DATA_DIR}/{item}.txt") as fin:
                for line in fin:

                    h, r, t = line.strip().split()
                    h, r, t = entity2id[h], relation2id[r], entity2id[t]

                    edges.append([h, r, t])
                    edges.append([t, r + mov, h])
                    ret['T'][int(r)] += 1
                    ret['T'][int(r + mov)] += 1
                    ret['Rh'][int(r)].add(h)
                    ret['Rt'][int(r)].add(t)
                    ret['Rh'][int(r + mov)].add(t)
                    ret['Rt'][int(r + mov)].add(h)
        else:
            with open(f"{DATA_DIR}/{item}.txt") as fin:
                for line in fin:

                    h, r, t = line.strip().split()
                    h, r, t = entity2id[h], relation2id[r], entity2id[t]

                    edges.append([h, r, t])
                    edges.append([t, r + mov, h])


        ret[item] = edges

    return ret


def build_graph(edges, E, R):
    return Graph(edges, num_node=E, num_relation=R)


def dataset_graph(dataset, edges='train'):
    return Graph(dataset[edges], num_node=dataset['E'], num_relation=dataset['R'])


def list2mask(a, N):
    if isinstance(a, list):
        a = torch.LongTensor(a)
    m = torch.zeros(N).to(a.device).bool()
    m[a] = True
    return m


def mask2list(m):
    N = m.size(0)
    m = m.cuda()
    a = torch.arange(N).cuda()
    return a[m]

if __name__ == "__main__":
    DATA_EM_DIR          = "./data_em/FB15k-237" 
    DATA_DIR          = "./data/FB15k-237" 
    DATA_EM_DIR          = "./data_em/umls" 
    DATA_DIR          = "./data/umls-rotate" 
    import numpy
    dims = [50,100,200,500,1000,2000] # modify for datasets
    for dim in dims:
        entity_embed = torch.tensor(numpy.load(f"{DATA_EM_DIR}/RotatE_{str(dim)}/entity_embedding.npy"))
        relation_embed = torch.tensor(numpy.load(f"{DATA_EM_DIR}/RotatE_{str(dim)}/relation_embedding.npy"))
        print(entity_embed.shape,relation_embed.shape)

        with open(f'{DATA_DIR}/entity2id.txt') as fin:
            id2entity = dict()
            for line in fin:
                entity,eid = line.strip().split('\t')
                id2entity[int(eid)] = entity

        with open(f'{DATA_DIR}/relation2id.txt') as fin:
            id2relation = dict()
            for line in fin:
                relation,rid = line.strip().split('\t')
                id2relation[int(rid)] = relation

        with open(f'{DATA_EM_DIR}/entities.dict') as fin:
            entity2id_em = dict()
            for line in fin:
                eid,entity = line.strip().split('\t')
                entity2id_em[entity] = int(eid)

        with open(f'{DATA_EM_DIR}/relations.dict') as fin:
            relation2id_em = dict()
            for line in fin:
                rid,relation = line.strip().split('\t')
                relation2id_em[relation] = int(rid)
        trE = dict()#id2emid
        trR = dict()
        elist = []
        rlist = []


        for eid in range(len(id2entity)):
            neid = entity2id_em[id2entity[eid]]
            trE[eid] = neid#id2emid
            elist.append(neid)#id2emid
        for rid in range(len(id2relation)):
            nrid = relation2id_em[id2relation[rid]]
            trR[rid] = nrid
            rlist.append(nrid)

        n_entity_embed = entity_embed[elist]
        n_relation_embed = relation_embed[rlist]
        numpy.save(f"{DATA_DIR}/RotatE_{str(dim)}/entity_embedding.npy",n_entity_embed)
        numpy.save(f"{DATA_DIR}/RotatE_{str(dim)}/relation_embedding.npy",n_relation_embed)
