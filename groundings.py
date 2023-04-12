import groundings_cppext as cppgnd
import torch

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
        print("groundings out")
        return {k: v for k, v in zip(key, val)}
    else:
        cppgnd.calc(h, rule)
        # print("groundings out")
        return cppgnd.result_pts()#一系列实体id，表示路径的终点
