import groundings_cppext as cppgnd
import torch
from att import *


def init_groundings(Noload = True):
    file = args.data + "/cb_Corpus_graph.pickle"
    if not os.path.exists(file):
        kg = Corpus_.get_multiroute_graph()
        file = args.data + "/Corpus_graph.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(kg, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading Generated graph  >>>")
        #cb_kg = pickle.load(open(args.data + "/cb_Corpus_graph.pickle",'rb'))
        kg = pickle.load(open(args.data + "/Corpus_graph.pickle",'rb'))

    file = args.data +"/groundings.pickle"#ground[head][path]={grounding:count,...}
    if (not os.path.exists(file)) or Noload:
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
    else:
        if os.path.getsize(file) > 0:
            print("Loading Generated groundings  >>>")
            ground = pickle.load(open(file,'rb'))
    return ground

def update_groundings(new_ground):
    file = args.data +"/groundings.pickle"#ground[head][path]=[grounding_list]
    with open(file, 'wb') as handle:
        pickle.dump(new_ground, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

def use_graph(g):
    cppgnd.init(g.num_node, g.num_relation)
    for e in g.edge_list:
        h, r, t = e.cpu().numpy().tolist()
        cppgnd.add(h, r, t)

def groundings(h, rule, count=False):
    # print("groundings in")
    if not isinstance(rule, list):
        if isinstance(rule, torch.Tensor):
            rule = rule.cpu().numpy().tolist()
        else:
            rule = list(rule)
    if count:
        cppgnd.calc_count(h, rule)
        key = cppgnd.result_pts()
        val = cppgnd.result_cnt()

        return {k: v for k, v in zip(key, val)}
    else:
        cppgnd.calc(h, rule)
        # print("groundings out")
        return cppgnd.result_pts()
