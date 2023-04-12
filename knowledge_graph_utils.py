import torch

from extra.data.graph import Graph


def load_dataset(DATA_DIR):
    entity2id = dict()
    relation2id = dict()

    with open(f'{DATA_DIR}/entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(f'{DATA_DIR}/relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    E = len(entity2id)
    R = len(relation2id)

    mov = R
    R += mov
    # R += 1

    ret = dict()
    ret['E'] = E#实体数
    ret['R'] = R#关系数，包括逆关系

    for item in ['train', 'valid', 'test']:
        edges = []
        with open(f"{DATA_DIR}/{item}.txt") as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                h, r, t = entity2id[h], relation2id[r], entity2id[t]

                edges.append([h, r, t])
                edges.append([t, r + mov, h])
        ret[item] = edges

    return ret#键值为['train', 'valid', 'test']的字典，每个字典的key是三元组(h,r,t)，及反三元组(t,r+mov,h)=的list。将r加上位移表示其逆关系，用于无向图？


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
