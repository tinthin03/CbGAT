import groundings_cppext as cppgnd
import torch
from att import *

def init_groundings():#来自文件，需要与use_graph的数据集保持一致
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

    file = args.data +"/groundings.pickle"#ground[head][path]={grounding:count,...}
    if not os.path.exists(file):
        ground = dict()
        for e in range(len(Corpus_.entity2id)):
            ground[e]=dict()
        for h,t_dict in kg.items():
            for t,r_list in t_dict.items():
                for r in r_list:
                    key = str(r)
                    if key in ground[h].keys():
                        if t not in ground[h][key].keys():
                            ground[h][key][t] = 1
                        else:
                            ground[h][key][t] += 1
                    else:
                        ground[h][key] = {t:1}
        with open(file, 'wb') as handle:
            pickle.dump(ground, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated groundings  >>>")
        ground = pickle.load(open(file,'rb'))
    return ground

def update_groundings(new_ground):
    file = args.data +"/groundings.pickle"#ground[head][path]=[grounding_list]
    with open(file, 'wb') as handle:
        pickle.dump(new_ground, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

#调用Groundings.cpp里的graph对象，形成e[h][r].push_back(t),a[h].push_back({r, t})两个数组表示知识，存入G.e,G.a
def use_graph(g):
    cppgnd.init(g.num_node, g.num_relation)
    for e in g.edge_list:
        h, r, t = e.cpu().numpy().tolist()
        cppgnd.add(h, r, t)

#返回head出发，通过规则路径得到的实体。（利用c++）
def groundings(h, rule, count=False):
    # print("groundings in")
    if not isinstance(rule, list):#rule转为list
        if isinstance(rule, torch.Tensor):
            rule = rule.cpu().numpy().tolist()
        else:
            rule = list(rule)
    if count:
        cppgnd.calc_count(h, rule)
        key = cppgnd.result_pts()
        val = cppgnd.result_cnt()
        # return list(zip(key, val))
        #print("groundings out")
        return {k: v for k, v in zip(key, val)}
    else:
        cppgnd.calc(h, rule)
        # print("groundings out")
        return cppgnd.result_pts()#一系列实体id，表示路径的终点
