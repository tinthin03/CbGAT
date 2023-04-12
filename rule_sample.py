import torch
import rule_sample_cppext as cpprs
#cppext_rule_sample是c++的namespace
#传入graph类的对象（一般是训练集的三元组样本构建），把三元组样本读入c++
def use_graph(g):
    cpprs.init(g.num_node, g.num_relation)
    for e in g.edge_list:
        h, r, t = e.cpu().numpy().tolist()
        cpprs.add(h, r, t)#实际上调用了c++的Graph结构体的add方法，形成2个二维向量来存放三元组。见grounding.cpp

#对r，抽样若干可能路径作为规则，length_time表示每个三元组长度规则的抽样数目,对每个r抽样用到num_samples个三元组(默认1000个)
def sample(r, length_time, num_samples, num_threads=1, samples_per_print=1):
    return cpprs.run(r, length_time, num_samples, num_threads, samples_per_print)


def save(rules, file):
    with open(file, 'w') as fin:
        for rule, (prec, recall) in rules:
            rule = ' '.join(map(str, rule))
            fin.write(f"{rule}\t{prec} {recall}\n")
