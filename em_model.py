import copy
import gc
from collections import defaultdict
import torch

import groundings
from knowledge_graph_utils import mask2list, list2mask, build_graph
from metrics import Metrics
from reasoning_model import ReasoningModel
from rotate import RotatE
from att import *

cos = torch.nn.CosineSimilarity(dim=0,eps=1e-12)#(1,-1)eps小一些，可以体高精度
#注意，每次调用train_Model，都需要指定一个目标谓词

#暂存groundings；其知识来自文件，需要与use_graph的数据集保持一致
session_groundings = groundings.init_groundings()

def calc_groundings(h, rule,count=False):
    path = ""
    for r in rule:
        path = path+str(r)+' '
    path = path[:-1]
    if path in session_groundings[h].keys():
        if not count:
            rgnd = list(session_groundings[h][path].keys())
        else:
            rgnd = session_groundings[h][path]
    else:
        rgnd = groundings.groundings(h, rule,count)
        session_groundings[h][path] = dict()
        if not count:
            for t in rgnd:
                if t not in session_groundings[h][path].keys():
                    session_groundings[h][path].update({t:1})
        else:
            #for t,cnt in rgnd
            session_groundings[h][path].update(rgnd)
    return rgnd

class EMiner(torch.nn.Module):
    def __init__(self, dataset, args, print=print):
        super(EMiner, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R']
        assert self.R % 2 == 0
        args = self.set_args(args)
        self.result = []

        self.dataset = dataset
        self._print = print
        self.print = self.log_print

        self.predictor_init = lambda: Evaluator(self.dataset,self._args, print=self.log_print)
        self.generator = CbGATGenerator(self.R, self.arg('generator_embed_dim'), self.arg('generator_hidden_dim'),self.dataset['Rh'],self.dataset['Rt'],
                                           print=self.log_print)

        print = self.print
        print("EMiner Init,self.E, self.R:", self.E, self.R)#14541 474

    def log_print(self, *args, **kwargs):
        import datetime
        timestr = datetime.datetime.now().strftime("%H:%M:%S.%f")
        if hasattr(self, 'em'):
            emstr = self.em if self.em < self.num_em_epoch else '#'
            prefix = f"r = {self.r} EM = {emstr}"
        else:
            prefix = "init"
        self._print(f"[{timestr}] {prefix} | ", end="")
        self._print(*args, **kwargs)

    # Use EM algorithm to train EMiner model
    #针对r，训练EMiner
    #rule_file是采用groudings初始化的规则。保护规则体和可信度：[rule path],{prec,rec}
    def train_model(self, r, num_em_epoch=None, rule_file=None, model_file=None):
        if rule_file is None:
            rule_file = f"rules_{r}.txt"
        if model_file is None:
            model_file = f"model_{r}.pth"
        if num_em_epoch is None:
            num_em_epoch = self.arg('num_em_epoch')

        self.num_em_epoch = num_em_epoch
        self.r = r  
        print = self.print

        pgnd_buffer = dict()
        rgnd_buffer = dict()
        rgnd_buffer_test = dict()
        self.generator.init()
        max_beam_rules = 3000
        def generate_rules():#生成当前目标谓词r的rule并初始化评估器
            if self.em == 0:#当前目标谓词r的第一次运行，完全init，此时使用事先生成的rule_file
                print("Use rule file to init.")
                self.predictor.relation_init(r=r, rule_file=rule_file, force_init_weight=self.arg('init_weight_boot'))
            else:
                sampled = set()
                sampled.add((r,))
                sampled.add(tuple())

                rules = [(r,)]#目标谓词r
                prior = [0.0, ]
                grules, gscores = self.generator.rule_gen(r,rule_file,self.arg('max_beam_rules'),self.predictor.arg('max_rule_len'))
                i = 0
                for rule in grules:
                    score = gscores[i].item()
                    rule = tuple(rule)
                    if rule in sampled:
                        i += 1
                        continue
                    sampled.add(rule)
                    rules.append(rule)
                    prior.append(score)
                    if len(sampled) % self.arg('sample_print_epoch') == 0 or len(sampled) < 20:
                        print(f"sampled # = {len(sampled)} rule = {rule} prior-score = {score} prec={self.generator.prec[i]}")
                    i += 1

                print(f"Done |sampled| = {len(sampled)}")

                prior = torch.tensor(prior).cuda()
                # prior -= prior.max()
                # prior = prior.exp()

                self.predictor.relation_init(r, rules=rules,path_gnd = self.generator.path_gnd,prec = self.generator.prec, prior=prior)#对每个新的EM epoch，初始化评估器，但不再使用rule_file
                self.generator.path_gnd = self.predictor.path_gnd
                self.generator.prec = self.predictor.prec
                self.generator.rules_path = self.predictor.rules_exp#经历过一次筛选的（可能是predictor，也可能直接用生成器的prior）

        for self.em in range(num_em_epoch):
            self.predictor = self.predictor_init()
            self.predictor.pgnd_buffer = pgnd_buffer
            self.predictor.rgnd_buffer = rgnd_buffer
            self.predictor.rgnd_buffer_test = rgnd_buffer_test
            self.predictor.em = self.em

            self.generator.r = self.r
            
            
            generate_rules()#生成当前目标谓词r的rule并初始化评估器（规则经历两次筛选，第一次生成器，第二次根据参数决定，直接用生成器的prior或使用predictor）

            # E-Step:
            print("Train/test Evaluator(E_step).")
            valid, test,ret_loss,ret_loss_rule = self.predictor.train_model()#执行规则评估器的训练，设置了self.training为真

            # M-Step
            #gen_batch = self.predictor.make_gen_batch()#用当前评估器模型生成规则数据作为rnn生成器的训练数据集，包括input（目标谓词和占位词序列），target（规则序列），weight，对应规则序列的权重
            if self.em>1:#TODO
                print("Train generator(M_step).")
                self.generator.train_model(r,ret_loss,ret_loss_rule,
                                        em_epoch=self.em,num_em_epoch = num_em_epoch)

            ckpt = {
                'r': r,
                'metrics': {
                    'valid': valid,
                    'test': test
                },
                'args': self._args_init,
                'rules': self.predictor.rules_exp,
                'predictor': self.state_dict(),
            }
            torch.save(ckpt, model_file)
            gc.collect()

        # Testing
        self.em = num_em_epoch
        self.predictor.em = self.em
        generate_rules()
        valid, test,ret_loss,ret_loss_rule = self.predictor.train_model()
        self.result = self.predictor.result_sum[-1]
        groundings.update_groundings(session_groundings)
        gc.collect()
        return valid, test

    def arg(self, name, apply=None):
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    # Definitions for EM framework
    def set_args(self, args):
        self._args_init = args
        self._args = dict()
        def_args = dict()
        def_args['num_em_epoch'] = 10
        def_args['sample_print_epoch'] = 20
        def_args['max_beam_rules'] = 3000#生成器生成或rulefile筛选的规则总数
        def_args['generator_embed_dim'] = 512
        def_args['generator_hidden_dim'] = 256
        def_args['generator_lr'] = 1e-3
        def_args['generator_num_epoch'] = 10000
        def_args['generator_print_epoch'] = 100
        def_args['init_weight_boot'] = False

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in def_args.items():
            self._args[k] = str(v) if make_str(v) else v
        for k, v in args.items():
            # if k not in self._args:
            # 	print(f"Warning: Unused argument '{k}'")
            self._args[k] = str(v) if make_str(v) else v


class Evaluator(ReasoningModel):
    def __init__(self, dataset, args, print=print):
        super(Evaluator, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R'] + 1 #237*2+1
        assert self.R % 2 == 1
        self.dataset = dataset
        #self.session_groundings = groundings

        self.set_args(args)
        rotate_pretrained = self.arg('rotate_pretrained', apply=lambda x: x)
        self.rotate = RotatE(dataset, rotate_pretrained)
        self.training = True

        self.rule_weight_raw = torch.nn.Parameter(torch.zeros(1))
        if rotate_pretrained is not None:
            if self.arg('param_relation_embed'):
                self.rotate.enable_parameter('relation_embed')
            if self.arg('param_entity_embed'):
                self.rotate.enable_parameter('entity_embed')

        self.pgnd_buffer = dict()
        self.rgnd_buffer = dict()
        self.rgnd_buffer_test = dict()
        self.cuda()
        self.print = print
        self.debug = False
        self.recording = False
        self.em = 0
        self.result_sum = []#[[0.0,0,0,0,0]]
        self.t_list = []#temp store t_list in train_step
        self.train_print = False

    def train(self, mode=True):
        self.training = mode
        super(Evaluator, self).train(mode)

    def eval(self):
        self.train(False)

    def index_select(self, tensor, index):
        if self.training:
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index)
            index = index.to(tensor.device)
            return tensor.index_select(0, index).squeeze(0)
        else:
            return tensor[index]

    @staticmethod
    def load_batch(batch):
        return tuple(map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, batch))

    # Fetch rule embeddings, either from buffer or by re-calculating
    #其取了rotate生成的关系表征相加，表示一个规则的表征
    def rule_embed(self, force=False):
        if not force and not self.arg('param_relation_embed'):
            return self._rule_embed

        relation_embed = self.rotate._attatch_empty_relation()
        rule_embed = torch.zeros(self.num_rule, self.rotate.embed_dim).cuda()
        for i in range(self.MAX_RULE_LEN):
            rule_embed += self.index_select(relation_embed, self.rules[i])#self.rules是规则序号的转置，第i列代表第所有规则的第i个谓词关系
        return rule_embed#shape = (self.num_rule, self.rotate.embed_dim)

    # Init rules#规则的预处理操作,包括设置当前评估器的（候选）规则，生成padding等
    #将num_rules置为rules的长度,
    #每epoch的relation_init中调用，最多生成max_beam_rules个rules
    def set_rules(self, rules):
        paths = rules
        r = self.r
        self.eval()

        # self.MAX_RULE_LEN = 0
        # for path in rules:
        # 	self.MAX_RULE_LEN = max(self.MAX_RULE_LEN, len(path))
        self.MAX_RULE_LEN = self.arg('max_rule_len')

        pad = self.R - 1
        gen_end = pad
        gen_pad = self.R
        rules = []
        rules_gen = []
        rules_exp = []
        #self.rules_path = rules

        for path in paths:
            npad = (self.MAX_RULE_LEN - len(path))
            rules.append(path + (pad,) * npad)
            rules_gen.append((r,) + path + (gen_end,) + (gen_pad,) * npad)
            rules_exp.append(tuple(path))

        self.rules = torch.LongTensor(rules).t().cuda()
        # print(self.rules.size())
        self.rules_gen = torch.LongTensor(rules_gen).cuda()
        self.rules_exp = tuple(rules_exp)

    @property
    def num_rule(self):
        return self.rules.size(1)

    # Finding pseudo-groundings for a specific (h, r)
    ##head实体，当前规则序号i，num，评估器当前的规则推理结果rgnd，寻找其中的伪grounding
    #根据head，规则的rotateE表征，根据rotateE的算法计算其loss（hOr-t），寻找使用规则i情况下，rotateE判断的tail
    #存入pgnd_buffer，并将使用规则i时rotateE判断的tail+已知grounding的tail实体返回
    def pgnd(self, h, i, num=None, rgnd=None):
        if num is None:
            num = self.arg('pgnd_num')

        key = (h, self.r, tuple(self.rules_exp[i]))
        if key in self.pgnd_buffer:
            return self.pgnd_buffer[key]#pgnd_buffer里有结果，则直接返回

        with torch.no_grad():
            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()
            ##规则的嵌入，tmp__rule_embed是直接把rotateE的关系嵌入相加
            rule_embed = self.rotate.embed(h, self.tmp__rule_embed[i])#根据rotateE算法计算得到的表征hOr
            if ans_can is None:
                dist = self.rotate.dist(rule_embed, self.rotate.entity_embed)#一个torch的function类，对所有实体，计算其与规则推理的tail的相似度距离
            else:
                can_dist = self.rotate.dist(rule_embed, self.rotate.entity_embed[ans_can])#已知答案，直接计算规则推理的tail与答案的相似度距离
                dist = torch.zeros(self.E).cuda() + 1e10
                dist[ans_can] = can_dist#答案跟规则推理结果的距离

            if rgnd is not None:
                # print(len(rgnd), dist.size())
                dist[torch.LongTensor(rgnd).cuda()] = 1e10#已知grounding的实体，将其距离置为0
            ret = torch.arange(self.E).cuda()[dist <= self.rotate.gamma]#筛选使用规则i情况下，rotateE判断的tail，因此就包含了一些伪grounding

            dist[ret] = 1e10#使用规则i时rotateE判断的tail+已知grounding的tail实体，其相似度距离都置为0
            num = min(num, dist.size(0) - len(rgnd)) - ret.size(-1)#min确保ret补的实体数小于非grounding的实体数
            if num > 0:#若ret的数量少于num，则从剩下的里面选取相似度接近tail的实体，补全之
                tmp = dist.topk(num, dim=0, largest=False, sorted=False)[1]
                ret = torch.cat([ret, tmp], dim=0)

        self.pgnd_buffer[key] = ret#ret：使用规则i时rotateE判断的tail+已知grounding的tail实体
        ##########
        # print(h, sorted(ret.cpu().numpy().tolist()))
        return ret

    # Calculate score in formula 17. A sparse matrix is given with column_idx=crule, row_idx=centity. Returns score in (17) in paper, as the value of the sparse matrix.
    #每个三元组均调用一次，利用rotate模型，计算各个batch（对应一个tripple）中：
    #（生成器的）规则（向量存于rule_embed、索引存于crule）的推断hOr，与当前（评估器的）pgnd给出的tail判断centity之间的相似度
    # 在运行过程中rule_embed代表的规则可能来自本轮生成器的生成，crule、centity代表的pgnd是当前评估器已存的规则（上一轮的规则），所以这两者的不同即为评估器的反馈 
    def cscore(self, rule_embed, crule, centity, cweight):
        # 针对当前三元组，batch中包含当前评估器下各个规则判断的每个tail实体（即pgnd的数量，等于 crule, centity的长度），这些对tail的判断存于centity，对应的规则存于crule
        # 本函数比较rule_embed（根据生成器给出规则结算得到的hOr,长度为num_rules等规则数量，为规则向量词典，不是pgnd数量），以及评估器中的pgnd的实体（t）对应的欧式距离相似度（用于判断哪些是tail），返回一个长度为pgnd的数量的数组
        # 其中生成器给出规则的hOr由crule索引，评估器中的pgnd的实体（t）由centity索引
        score = self.rotate.compare(rule_embed, self.rotate.entity_embed, crule, centity)
        score = (self.rotate.gamma - score).sigmoid()
        if self.arg('drop_neg_gnd'):
            score = score * (score >= 0.5)
        score = score * cweight#cweight在make_batch里初始化为ones * self.arg('pgnd_weight')
        return score#返回的是batch中各个规则，及其对应pgnd(通过rotateE判断，或者grounding方式判断)，之间的匹配度得分。

    # Returns the rule's value in (16)
    #对每个三元组对应的样本：h, t_list, mask, crule, centity, cweight
    #根据make_batch里centity实体，即根据当前评估器对tail的判断（=当前评估器规则在rotateE下的判断+规则的grounding的判断）而筛选出的tail候选
    #mask为centity中正确答案的掩码，正确答案由t_list或给定的answer决定
    #来评估rule，计算查询每个（h,r,?）下某个rule（存于crule）的概率权重。规则的数量为self.num_rule，已在setrule()函数置为rulefile或者生成器给出的规则数。
    #评估时，每个规则的正确预测的pgnd的cscore加和为pos，错误预测的的pgnd的cscore加和为neg，再求其差，除以规则各自的pgnd之和num
    # 在运行过程中self.tmp__rule_embed的规则可能来自本轮生成器的生成，crule, centity, cweight里的pgnd是当前评估器已存的规则（上一轮的规则），所以这两者的不同即为评估器的反馈 
    #这里实际上是对self.tmp__rule_value的学习过程，因为每个三元组都要执行一次，对self.tmp__rule_embed存储的各个规则进行评价
    def rule_value(self, batch, weighted=False):
        num_rule = self.num_rule
        h, t_list, mask, crule, centity, cweight = self.load_batch(batch)
        # print("rule_value--->h,t_list",h,t_list)
        with torch.no_grad():

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed)#hOt，可能来自本轮生成器的生成
            #（生成器的）规则（向量存于rule_embed、索引存于crule）的推断hOr，与当前（评估器的）pgnd给出的tail判断centity之间的相似度
            cscore = self.cscore(rule_embed, crule, centity, cweight)
            # print("cscore",cscore.shape,cscore)#torch.Size([55010]),长度等于pgnd的数量
            indices = torch.stack([crule, centity], 0)

            def cvalue(cscore):
                if cscore.size(0) == 0:
                    return torch.zeros(num_rule).cuda()
                return torch.sparse.sum(torch.sparse.FloatTensor(
                    indices,
                    cscore,
                    torch.Size([num_rule, self.E])#统计各个规则的得分之和，对同一规则的各个pgnd得分做加和
                ).cuda(), -1).to_dense()


            pos = cvalue(cscore * mask[centity])#该样本的各个规则的正得分
            neg = cvalue(cscore * ~mask[centity])#该样本的各个规则的负得分
            score = cvalue(cscore)
            num = cvalue(cweight).clamp(min=0.001)#加权和的分母

            pos_num = cvalue(cweight * mask[centity]).clamp(min=0.001)
            neg_num = cvalue(cweight * ~mask[centity]).clamp(min=0.001)


            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)
            #取参数'rule_value_def'同名的项（可能是pos、neg等）为规则的打分。默认：(pos - neg) / num
            value = self.arg('rule_value_def', apply=eval_ctx(locals()))
            # print("pos,neg,num",pos,neg,num)
            # print("value",value.shape,value)#torch.Size([3000])
            if weighted:
                value *= len(t_list)#对规则评估值乘以实际的tail数，表示其权重

            if hasattr(self, 'tmp__rule_value'):
                self.tmp__rule_value += value#
                self.tmp__num_init += len(t_list)

        return value

    # Choose rules, which has top `num_samples` of `value` and has a non negative `nonneg`
    #返回利用value（当前评估器的估值和生成器权重prior之和）排序的topK个候选规则。num_samples，最大pgnd数，默认为max_rules=1000
    #num_samples不为None时，为EM初始化的过程，此时生成的是E-step训练评估器的样本，数量为num_samples。num_samples是None时，选择的是生成器的样本
    def choose_rules(self, value, nonneg=None, num_samples=None, return_mask=False):
        if num_samples is None:#每个epoch的E-step之前，默认值为max_rules=1000
            num_samples = self.arg('max_best_rules')#M-step时使用，最大不超过max_best_rules个候选规则，默认300
        ################
        # print(f"choose_rules num = {num_samples}")
        with torch.no_grad():
            num_rule = value.size(-1)#候选规则的数量num_rules，此处已置为生成器、rulefile给出的规则数量
            topk = value.topk(min(num_samples - 1, num_rule), dim=0, largest=True, sorted=False)[1]
            cho = torch.zeros(num_rule).bool().cuda()#num_samples一般小于num_rule
            cho[topk] = True
            if nonneg is not None:
                cho[nonneg < 0] = False

        if return_mask:
            return cho
        return mask2list(cho)

    # Choose best rules for each batch, for M-step
    ##返回利用当前self.rule_value排序的topK = max_best_rules个候选规则
    def best_rules(self, batch, num_samples=None):
        with torch.no_grad():
            w = self.rule_value(batch)
            value = (w + self.arg('prior_coef') * self.prior) * self.rule_weight
            cho = self.choose_rules(value, nonneg=w, num_samples=num_samples)
        return cho

    # For a new relation, init rule weights and choose rules
    #对某个r，设置评估器的规则为rule_file事先生成的规则（第一次EM epoch）或生成器的规则（此后训练过程中）
    # 第一EM epoch初始化时，一定调用make_batch()，生成train_batch、test_batch等。
    # batch格式：h, t_list, list2mask(answer, self.E), crule, centity, cweight
    # 其中t_list为train集的tail实体，centity给出评估器当前规则下的rotateE算法选择的tail，以及grounding的tail
    #init_weight_with_prior默认为false，此时，每个epoch都会调用make_batch()来生成上述的batch，以更新评估器当前规则。
    def relation_init(self, r=None, rule_file=None,path_gnd = [],prec = [], rules=None, prior=None, force_init_weight=False):
        print = self.print
        if r is not None:
            self.r = r
        r = self.r
        if rules is None:#模型的第一次初始化
            assert rule_file is not None#模型的第一次初始化，使用rule_sample.cpp采样的规则初始化第一组样本集
            #rule_sample对每个谓词r，抽样若干可能路径作为规则，length_time表示每个三元组各长度规则的抽样数目,对每个目标规则r抽样pgnd，用到num_samples个规则(默认1000个)
            #并评估规则的先验权重，并存入rules_*文件
            rules = [((r,), 1, -1)]
            rule_set = set([tuple(), (r,)])
            has_inv = False
            with open(rule_file) as file:
                for i, line in enumerate(file):
                    try:
                        path, prec = line.split('\t')
                        path = tuple(map(int, path.split()))
                        prec = float(prec.split()[0])#采用rulesample的grpundingtruth评估结果prec作为初始的规则权重

                        if not (prec >= 0.0001):
                            # to avoid negative and nan
                            prec = 0.0001

                        if path in rule_set:
                            continue
                        for rel in path:
                            if rel >= (self.R-1)/2:
                                has_inv = True
                                break
                        if has_inv:
                            has_inv = False
                            continue#去掉rulefile里的逆关系规则
                        rule_set.add(path)
                        if len(path) <= self.arg('max_rule_len'):
                            rules.append((path, prec, i))
                    except:
                        continue

            rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)[:self.arg('max_beam_rules')]#利用权重排序rule
            print(f"Loaded from file: |rules| = {len(rules)} max_rule_len = {self.arg('max_rule_len')}")
            x = torch.tensor([prec for _, prec, _ in rules]).cuda()
            prior = -torch.log((1 - x.float()).clamp(min=1e-6))
            # prior = x
            rules = [path for path, _, _ in rules]
        else:#模型中间过程的初始化，直接采用生成器产生的规则作为评估器的规则
            assert prior is not None

        self.prior = prior#rule_sample给出的规则的初始权重，或者上一轮的生成器给出的规则权重
        self.set_rules(rules)#规则的预处理操作，设置评估器当前的规则为rules，生成padding等。此时即将self.rule_exp置为上一轮生成器产出
        self.path_gnd = path_gnd
        self.prec = prec
        print("Generator exctract max_beam_rules/num_rule.|rules|:",self.num_rule)
        num_rule = self.num_rule#已由self.set_rules函数置为生成器或者rulefile给出的规则数（默认为3000）,规则按照prior排序
        with torch.no_grad():
            self.tmp__rule_value = torch.zeros(num_rule).cuda()
            self.tmp__rule_embed = self.rule_embed(force=True).detach()#取了rotate生成的关系表征相加，表示一个规则的表征
            self.tmp__num_init = 0
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        init_weight = force_init_weight or not self.arg('init_weight_with_prior')
        if self.arg('rely_gen'):#依赖生成器规则，则value只考虑prior，不执行rule_value，self.tmp__rule_value一直为0
            init_weight = not self.arg('rely_gen') #False
        if init_weight:#初始化时看force_init_weight（默认为false），后续根据self.arg('init_weight_with_prior')的设置调用
            #make_batchs生成r的batch的格式：h, t_list, list2mask(answer, self.E), crule, centity, cweight
            #t_list为训练集给出的tail,centity为评估器当前的规则self.rules_exp（来自rule_file或生成器）下对tail的判断=使用规则i时rotateE判断的tail+已知grounding的tail实体
            for batch in self.make_batchs(init=True):#（init=True时，从训练集中生成batch）制造用于生成器的规则样本,样本数量等于三元组的数量；对每个batch，根据h、规则做grounding和rotateE计算pgnd
                #对每个三元组对应的样本，对当前self.rule_exp的前max_rules个规则做打分
                # 对每个给定规则，利用rotateE算法的hOr和规则的pgnd，根据正确答案t_list（或answer给出），给出规则的confidence
                #（self.rule_exp可能来自生成器/rule_file，此处是初始化故来自predicate从rule_file里初始得到的rule）
                self.rule_value(batch, weighted=True)#比较batch中的正确答案t_list和规则判断pgnd，对max_beam_rules个规则打分

        with torch.no_grad():
            # self.tmp__rule_value[torch.isnan(self.tmp__rule_value)] = 0
            #self.tmp__rule_value存评估器对本轮生成器规则的打分。长度为候选规则的数量num_rules.
            #tmp__num_init表示目标谓词r相关的t_list的总和，
            #self.prior：初始化为rule_sample给出的规则的初始权重，或者上一轮的生成器给出的规则权重
            avg_rule_value = self.tmp__rule_value / max(self.tmp__num_init, 1)
            print("Total tmp__rule_value,total avg(tmp__rule_value),self.prior",self.tmp__rule_value.sum(),avg_rule_value.sum(),self.prior.sum())
            # print(self.tmp__rule_value.shape,self.tmp__rule_value[0:10])#size = 3000 = max_beam_rules/num_rule
            # print(self.tmp__num_init)
            print(self.prior.shape,self.prior[0:50])#size = 3000 = max_beam_rules/num_rule
            value =  avg_rule_value + self.arg('prior_coef') * self.prior
            nonneg = self.tmp__rule_value
            if self.arg('use_neg_rules') or not init_weight:
                nonneg = None
            #value=评估器对本轮生成器规则的打分+self.prior（rule_sample给出的规则的初始权重，或者上一轮的生成器给出的规则权重），value默认应该是生成器或rulefile给出的规则数3000
            #根据最新的value，选择num_samples个新的规则作为训练评估器的样本。return_mask=True，因此cho在此处是长跟value一致的mask。默认样本数上限num_samples为max_rules=1000
            #评估器评分value，是考虑了真正答案t_list或answer的
            cho = self.choose_rules(value, num_samples=self.arg('max_rules'), nonneg=nonneg, return_mask=True)
            print("choose max_rules,update rule_weight_raw for Evaluator training.")
            cho[0] = True
            cho_list = mask2list(cho).detach().cpu().numpy().tolist()#吧cho由mask变为mask指向的序号list
            value_list = value.detach().cpu().numpy().tolist()
            cho_list = sorted(cho_list,
                              key=lambda x: (x == 0, value_list[x]), reverse=True)#规则序号按value中的值排序
            assert cho_list[0] == 0
            cho = torch.LongTensor(cho_list).cuda()

            value = value[cho]#epoch总打分按value大小排序，筛选max_rules个
            self.tmp__rule_value = self.tmp__rule_value[cho]#评估器的打分按value大小排序，筛选max_rules个
            self.prior = self.prior[cho]#本轮生成器打分，按value顺序排列，筛选max_rules个
            self.rules = self.rules[:, cho]
            self.rules_gen = self.rules_gen[cho]#
            self.rules_exp = [self.rules_exp[x] for x in cho_list]#self.rules_exp里的规则，按value顺序重新排列，筛选max_rules个
            #print("len(self.path_gnd),len(cho_list)",len(self.path_gnd),len(cho_list))
            if len(self.path_gnd)>0:
                self.path_gnd = [self.path_gnd[x] for x in cho_list]
                self.prec  = [self.prec[x] for x in cho_list]
        if init_weight:#init_weight_with_prior默认为false时，weight为tmp__rule_value
            weight = self.tmp__rule_value+self.prior*1000
        else:
            weight = self.prior#init_weight_with_prior设为true时，self.tmp__rule_value 一直为0，模型只接受prior为权重

        print(f"weight_init: Aft choose,|rules| = {self.num_rule} [{weight.min().item()}, {weight.max().item()}]")
        weight = weight.clamp(min=0.0001)
        weight /= weight.max()
        weight[0] = 1.0
        self.rule_weight_raw = torch.nn.Parameter(weight)#最终的评估、test，采用此权重给候选规则的pgnd加权

        del self.tmp__rule_value
        del self.tmp__num_init

        with torch.no_grad():
            self.tmp__rule_embed = self.rule_embed(force=True).detach()
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        self.make_batchs()#根据value筛选max_rules个规则，根据h、规则做grounding和rotateE计算pgnd，以更新评估器训练所使用的的样本
        print("Use max_rules,update pgnd for Evaluator training.")
        del self.tmp__rule_embed

    # Default arguments for predictor
    def set_args(self, args):
        self._args = dict()
        def_args = dict()
        def_args['rotate_pretrained'] = None
        def_args['max_beam_rules'] = 3000#生成器生成或rulefile筛选的规则总数
        def_args['max_rules'] = 1000##对每个h/样本batch抽取pgnd时，使用规则的最大数量（也是E-step训练评估器的样本数量）。如crule=[0,0,1., 1., 1., 2., 2., 2.]，则此数为3。
        def_args['max_rule_len'] = 4
        def_args['max_h'] = 5000
        def_args['max_best_rules'] = 300#每epoch评估器生成的生成器训练样本的最大数量，M-step的best_rules()时使用
        def_args['param_relation_embed'] = True
        def_args['param_entity_embed'] = False
        def_args['init_weight_with_prior'] = False#False时，每个epoch调用relation_init时，都使用当前的评估器规则来重新生成batch，基于当前规则下rotateE的判断和grounding来更新centity。init_weight_with_prior设为true时，self.tmp__rule_value 一直为0，则可以专注训练生成器
        def_args['prior_coef'] = 1000#0.01
        def_args['use_neg_rules'] = False
        def_args['disable_gnd'] = False#是否在生成每个样本的pgnd时，考虑grounding本身的判断结果为pgnd。为false时，不但考虑rotateE的判断，还考虑grounding
        def_args['disable_selflink'] = False
        def_args['drop_neg_gnd'] = False
        def_args['pgnd_num'] = 256
        def_args['pgnd_selflink_rate'] = 8
        def_args['pgnd_nonselflink_rate'] = 0
        def_args['pgnd_weight'] = 0.1
        def_args['max_pgnd_rules'] = None  # def_args['max_rules']
        def_args['predictor_num_epoch'] = 10000#200000
        def_args['predictor_early_break_rate'] = 1 / 5
        def_args['predictor_lr'] = 5e-5
        def_args['predictor_batch_size'] = 1
        def_args['predictor_print_epoch'] = 100
        def_args['predictor_init_print_epoch'] = 10
        def_args['predictor_valid_epoch'] = 100
        def_args['predictor_eval_rate'] = 4
        def_args['rule_value_def'] = '(pos - neg) / num'
        def_args['metrics_score_def'] = '(mrr+0.9*h1+0.8*h3+0.7*h10+0.01/max(1,mr), mrr, mr, h1, h3, h10, -mr)'
        def_args['answer_candidates'] = None
        def_args['record_test'] = False
        def_args['rely_gen'] = True

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in def_args.items():
            self._args[k] = str(v) if make_str(v) else v
        for k, v in args.items():
            # if k not in self._args:
            # 	print(f"Warning: Unused argument '{k}'")
            self._args[k] = str(v) if make_str(v) else v

    def arg(self, name, apply=None):
        # print(self._args[name])
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    @property
    def rule_weight(self):
        return self.rule_weight_raw

    def forward(self, batch,individual_rule = True):

        E = self.E
        R = self.R

        rule_weight = self.rule_weight
        _rule_embed = self.rule_embed()#shape = (self.num_rule, self.rotate.embed_dim)
        rule_embed = []
        crule = []
        crule_weight = []
        centity = []
        cweight = []
        csplit = [0]

        for single in batch:
            #每个batch对应一个三元组head，存了目标谓词r相关的score最大的前max_rules个规则。score根据评估器rotateE的参数，跟t_list正确答案作比较的分数+生成器的先验得分得出
            #crule是一个数组，长度为 r的各个规则当前推理的pgnd的数量，值是规则在集合里的序号。_centity, _cweight形状相同
            _h, _, _, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) == 0:
                csplit.append(csplit[-1])#空样本置跟上一个同样的值，这样的样本的head没有任何规则可推导出pgnd
                continue
            crule.append(_crule + len(rule_embed) * self.num_rule)#规则序号的编码，每batch里涉及的规则共self.num_rule个，因此下一行的序号增加行数*self.num_rule
            crule_weight.append(rule_weight.index_select(0, _crule))#len=pgnd
            centity.append(_centity)
            cweight.append(_cweight)
            rule_embed.append(self.rotate.embed(_h, _rule_embed))#hOr，代表num_rule个规则在rotatE模型下，对tail的表征的推理 ,(self.num_rule, self.rotate.embed_dim)
            csplit.append(csplit[-1] + _crule.size(-1))#规则序列里的占位分离符，记录该样本的pgnd的序号


        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            crule_weight = torch.tensor([]).cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).cuda()
            rule_embed = torch.tensor([]).cuda()
            cscore = torch.tensor([]).cuda()
            cscore_drule = torch.tensor([]).cuda()
        else:
            crule = torch.cat(crule, dim=0)#合并所有样本的pgnd。其中规则的序号为原始规则序号+偏移，偏移==样本序号*self.num_rule
            crule_weight = torch.cat(crule_weight, dim=0)#规则权重
            centity = torch.cat(centity, dim=0)#所有样本的pgnd
            cweight = torch.cat(cweight, dim=0)
            rule_embed = torch.cat(rule_embed, dim=0)#每个样本里num_rule个规则的推理结果hOr的叠加，len = self.num_rule*tripple样本数
            cscore_drule = self.cscore(rule_embed, crule, centity, cweight)#用于统计单rule的质量，规则序号crule经过偏移后跟rule_embed的序号正好对应，len = pgnd总量
            cscore = cscore_drule * crule_weight#所有样本的pgnd，都按评估器的评分算法打分,并以规则的分数加权，此分数不涉及答案t_list。len = pgnd总量

        loss = torch.tensor(0.0).cuda().requires_grad_() + 0.0#requires_grad_()将requires_grad置为true？？;真正可微的是cscore
        #loss_rule = torch.zeros(self.num_rule).cuda() + 0.0
        loss_rule = 0.0
        result = []
        augfix = 10000
        for i, single in enumerate(batch):#再遍历一遍三元组
            #mask标记了E个实体中的正确答案
            _h, t_list, mask, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) != 0:#head有规则可达pgnd
                crange = torch.arange(csplit[i], csplit[i + 1]).cuda()#截取某个样本对应的pgnd数据
                sparse_score = torch.sparse.FloatTensor(
                    torch.stack([_centity, _crule], dim=0),
                    self.index_select(cscore, crange),
                    torch.Size([E, self.num_rule])
                )
                score = torch.sparse.sum(sparse_score, -1).to_dense()#计算评估器基于该样本的规则和pgnd，对全部E个实体的打分;shape = (E,1)
                if individual_rule:
                    score_rule = torch.sparse.FloatTensor(
                    torch.stack([_centity, _crule], dim=0),
                    self.index_select(cscore_drule, crange),
                    torch.Size([E, self.num_rule])).to_dense()#shape = (E,num_rule)
            else:#head无规则可达pgnd，可能是KG上的末梢节点
                score = torch.zeros(self.E).cuda()
                score.requires_grad_()
                if individual_rule:
                    score_rule = torch.zeros(self.E,self.num_rule).cuda()
                    #score_rule.requires_grad_()

            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:#手动指定了answer，则加入t_list。一般是valid、test时用
                ans_can = ans_can.cuda()
                score = self.index_select(score, ans_can)
                
                if individual_rule:
                    #score_rule = score_rule.index_select(0, ans_can)
                    #score_rule = self.index_select(score_rule, ans_can)

                    if self.training:
                        if not isinstance(ans_can, torch.Tensor):
                            ans_can = torch.tensor(ans_can)
                        ans_can = ans_can.to(score_rule.device)
                        score_rule = score_rule.index_select(0, ans_can)
                    else:
                        score_rule = score_rule[ans_can]

                mask = self.index_select(mask, ans_can)

                map_arr = -torch.ones(self.E).long().cuda()
                map_arr[ans_can] = torch.arange(ans_can.size(0)).long().cuda()
                map_arr = map_arr.detach().cpu().numpy().tolist()
                map_fn = lambda x: map_arr[x]
                t_list = list(map(map_fn, t_list))
                #根据ans_can的赋值（一个表示tail答案的一维tensor）,改变t_list的值：ans_can里出现的实体为0，其他的为-1。同时从score中筛选相应的答案的得分，筛选相应的mask

            if self.recording:
                self.record.append((score.cpu(), mask, t_list))

            elif not self.training:#valid时
                for t in t_list:
                    result.append(self.metrics.apply(score, mask.bool(), t))#valid计算hit@1、mrr等具体过程
            #训练时：t_list一般来自训练集
            if score.dim() == 0:
                continue

            score = score.softmax(dim=-1)#归一化得分,shape = (E,1)
            neg = score.masked_select(~mask.bool())#所有错误预测的score直接加和作为loss;shape = (E,1)

            loss += neg.sum()

            for t in t_list:#错误答案里，score高于正确答案的，再加和一次其score作为loss
                s = score[t]
                wrong = (neg > s)
                loss += ((neg - s) * wrong).sum() / wrong.sum().clamp(min=1)
            
            if individual_rule:

                with torch.no_grad():

                    sc_rule = score_rule#.softmax(dim=0)#shape = (E,num_rule)
                    mask_rule = mask.unsqueeze(-1).repeat(1,self.num_rule)
                    #print(mask_rule,mask_rule.shape)
                    #ng_rule = sc_rule.masked_select(~mask_rule.bool())
                    ng_rule = (sc_rule*~mask_rule)
                    #print("ng_rule",ng_rule,ng_rule.shape)
                    loss_rule += ng_rule.sum(0)*augfix
                    #print("loss_rule",loss_rule)
                    # if self.train_print:
                    #     print(f"For tail {t_list},{loss_rule.shape}rules,rule_score = {(sc_rule[t_list].sum(0)*augfix)[:10]}")


                    for t in t_list:
                        #s_rule = sc_rule[t]
                        #print(s_rule)
                        #wrong_rule = (ng_rule > sc_rule[t])
                        ls_ng =augfix*((ng_rule - sc_rule[t]) * (ng_rule > sc_rule[t])).sum(0) 
                        ls_f =  (ng_rule > sc_rule[t]).sum(0).clamp(min=1)
                        # print(wrong_rule)
                        # print(ng_rule - s_rule)
                        # print(((ng_rule - s_rule) * wrong_rule).sum(0))
                        # print(wrong_rule.sum(0))
                        loss_rule += ls_ng / ls_f#考虑不除以wrong_rule.sum(0)
                    # ls_ng =((ng_rule - sc_rule[t_list[0]]) * (ng_rule > sc_rule[t_list[0]])).sum(0) 
                    # ls_f =  (ng_rule > sc_rule[t_list[0]]).sum(0).clamp(min=1)
                    # loss_rule += ls_ng / ls_f
                    #print("total loss_rule",loss_rule)
        #loss_rule.shape = (self.num_rule)
        self.t_list = t_list
        return loss / len(batch),loss_rule/len(batch), self.metrics.summary(result)

    def _evaluate(self, valid_batch, batch_size=None):
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')#4
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')
        # print(print_epoch)

        self.eval()
        with torch.no_grad():
            result = Metrics.zero_value()
            for i in range(0, len(valid_batch), batch_size):#每4个batch进行一个valid
                cur = model(valid_batch[i: i + batch_size])[2]
                result = Metrics.merge(result, cur)
                if i % print_epoch == 0 and i > 0:
                    print(f"eval #{i}/{len(valid_batch)}")
        return result
    


    # make a single batch, find groundings and pseudo-groundings
    #根据当前目标谓词r，当前规则集self.rules_exp，head实体，推理尾实体作为grounding truth
    ##对每个h/样本batch抽取pgnd时，使用规则的最大数量为为max_rules，默认值为1000（不等于生成器给出的规则数num_rules），即每个batch只涉及max_rules个规则的评估
    def _make_batch(self, h, t_list, answer=None, rgnd_buffer=None):
        # print("make_batch in")
        if answer is None:
            answer = t_list
        if rgnd_buffer is None:
            rgnd_buffer = self.rgnd_buffer
        crule = []
        centity = []
        cweight = []
        gnd = []
        max_pgnd_rules = self.arg('max_pgnd_rules')
        if max_pgnd_rules is None:
            max_pgnd_rules = self.arg('max_rules')#默认为max_rules，值为1000
        for i, rule in enumerate(self.rules_exp):#遍历跟r相关的各个规则，rules_exp是rule path，此时类型为元祖，调用这个函数后，后面可能改为list
            # print(f"iter i = {i} / {len(self.rules_exp)}")
            if i != 0 and not self.arg('disable_gnd'):#disable_gnd默认为false，此时利用kg，给出rule的推理结果实体集
                key = (h, self.r, rule)
                if key in rgnd_buffer:
                    rgnd = rgnd_buffer[key]
                else:
                    # print("gnd in")
                    rgnd = calc_groundings(h, rule)#（利用c++）返回head出发，通过规则路径得到的终点实体id，list

                    ans_can = self.arg('answer_candidates', apply=lambda x: x)#人工给出候选答案
                    if ans_can is not None:
                        ans_can = set(ans_can.cpu().numpy().tolist())
                        rgnd = list(filter(lambda x: x in ans_can, rgnd))#加入人工给出的候选答案
                    rgnd_buffer[key] = rgnd#rgnd_buffer记录 (h, self.r, rule)对应的各个推理结果

                ones = torch.ones(len(rgnd))
                centity.append(torch.LongTensor(rgnd))#利用各个规则推理得到的结果实体list
                crule.append(ones.long() * i)#规则的序号，用于生成器的训练样本
                cweight.append(ones)#各个规则的初始权重，设为1
            else:
                rgnd = []

            gnd.append(rgnd)
            if i == 0 and self.arg('disable_selflink'):
                continue
            if i >= max_pgnd_rules:#对每个h/样本batch抽取pgnd时，使用规则的最大数量
                continue
            num = self.arg('pgnd_num') * self.arg('pgnd_selflink_rate' if i == 0 else 'pgnd_nonselflink_rate')
            #输入head，当前规则序号i，num，当前的规则推理结果rgnd，返回rgnd_buffer里当前规则的推理结果
            #返回使用规则i时rotateE判断的tail+已知grounding的tail实体（gnd[i]传入），数量至少是num个
            
            if self.arg('rely_gen'):pgnd = torch.LongTensor(gnd[i])#依赖生成器规则，仅考虑规则直接产生的groundings
            else:pgnd = self.pgnd(h, i, num, gnd[i])
            

            ones = torch.ones(len(pgnd))
            centity.append(pgnd.long().cpu())#当前规则下对tail的判断=使用规则i时rotateE判断的tail+已知grounding的tail实体
            crule.append(ones.long() * i)#规则的序号，长度为该规则判断的tail实体的数量
            cweight.append(ones * self.arg('pgnd_weight'))#置信tail的权重设置

        # print("iter done")
        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).float().cuda()
        else:
            crule = torch.cat(crule, dim=0)#长度为当前评估器下各个规则判断的tail实体的总数量，值为规则的序号
            centity = torch.cat(centity, dim=0)#当前评估器下各个规则判断的所有tail实体的拼接
            cweight = torch.cat(cweight, dim=0)#当前评估器下各规则的score，作为规则的权重（用于生产器训练）

        #################
        # print("work", answer)

        # print("make_batch out")

        return h, t_list, list2mask(answer, self.E), crule, centity, cweight

    # make all batchs
    #batch的格式：h, t_list, list2mask(answer, self.E), crule, centity, cweight
    #t_list为训练集给出的tail,centity为当前规则下对tail的判断=使用规则i时rotateE判断的tail+已知grounding的tail实体
    #注意，每个三元组对应上面的一行batch， crule, centity, cweight的长度等于当前预测器所有关于r的规则所判断的tail实体数量和（即pgnd的长度）
    def make_batchs(self, init=False):
        print = self.print
        # if r is not None:
        # 	self.r = r
        dataset = self.dataset
        graph = build_graph(dataset['train'], self.E, self.R)#利用训练集知识生成的Graph类的对象，仅训练拟合使用
        graph_test = build_graph(dataset['train'] + dataset['valid'], self.E, self.R)

        def filter(tri):
            a = defaultdict(lambda: [])
            for h, r, t in tri:
                if r == self.r:
                    a[h].append(t)
            return a

        train = filter(dataset['train'])#取出数据集中关系为self.r的三元组，表示为一个字典，k为head实体，值为tail实体组成的list
        valid = filter(dataset['valid'])
        test = filter(dataset['test'])

        answer_valid = defaultdict(lambda: [])
        answer_test = defaultdict(lambda: [])
        for a in [train, valid]:
            for k, v in a.items():
                answer_valid[k] += v#train和valid的tri合并
                answer_test[k] += v
        for k, v in test.items():
            answer_test[k] += v#train和valid、test的tri合并

        if len(train) > self.arg('max_h'):#随机抽取最大不超过max_h个训练样本？
            from random import shuffle
            train = list(train.items())
            shuffle(train)
            train = train[:self.arg('max_h')]
            train = {k: v for (k, v) in train}#关系为self.r的三元组，表示为一个字典，k为head实体，值为tail实体组成的list

        print_epoch = self.arg('predictor_init_print_epoch')

        self.train_batch = []
        self.valid_batch = []
        self.test_batch = []

        groundings.use_graph(graph)#生成train_batch前，只把train图加入groundings，不加valid图

        if init:
            def gen_init(self, train, print_epoch):
                for i, (h, t_list) in enumerate(train.items()):
                    if i % print_epoch == 0:
                        print(f"init_batch: {i}/{len(train)}")
                    yield self._make_batch(h, t_list)

            return gen_init(self, train, print_epoch)#初始时，从训练集生成这种batch，并返回batch
        #_make_batch：h, t_list, list2mask(answer, self.E), crule, centity, cweight
        #t_list：训练集给出的跟h，r对应的tail组成的list
        # 最重要的返回是centity，即当前规则下对tail的判断=使用规则i时rotateE判断的tail+已知grounding的tail实体
        #每个三元组都生成上述一行，作为样本
        for i, (h, t_list) in enumerate(train.items()):#遍历self.r关联的head实体及对应的tail的list
            if i % print_epoch == 0:
                print(f"train_batch: {i}/{len(train)}")
            batch = list(self._make_batch(h, t_list))
            for t in t_list:
                batch[1] = [t]
                self.train_batch.append(tuple(batch))#train_batch里，根据t_list拆成独立的头尾实体对，不再是h和t_list一对多

        for i, (h, t_list) in enumerate(valid.items()):#非初始化时，需要生成valid_batch，用train做grounding，answer_valid作为答案
            if i % print_epoch == 0:
                print(f"valid_batch: {i}/{len(valid)}")
            self.valid_batch.append(self._make_batch(h, t_list, answer=answer_valid[h]))

        groundings.use_graph(graph_test)#生成test_batch前，需要把train+valid图加入groundings
        for i, (h, t_list) in enumerate(test.items()):#非初始化时，需要生成test_batch
            if i % print_epoch == 0:
                print(f"test_batch: {i}/{len(test)}")
            self.test_batch.append(
                self._make_batch(h, t_list, answer=answer_test[h], rgnd_buffer=self.rgnd_buffer_test))

    # Make batchs for generator, used in M-step
    #抽样产生新的rules，返回值包括：
    #inputs：目标谓词，规则体（batchsize个定长一维list，长度为rulelenth+1，填充padding_index=num_relations + 1）；
	# target：样本里的规则体，终止id：ending_idx（shape为（batchsize,rulelenth+1)的tensor，填充padding_index=num_relations + 1）;
	# weight：Hscore;
    def make_gen_batch(self, generator_version=1):
        self.tmp__rule_embed = self.rule_embed()##其取了rotate生成的关系表征相加，表示一个规则的表征
        weight = torch.zeros_like(self.rule_weight_raw).long().cuda()#讲权重初始化为0
        for i, batch in enumerate(self.train_batch):#评估器利用训练集给出评估，获取当前的top max_best_rules规则，并赋值权重
            cho = self.best_rules(batch)#每个batch，选取max_best_rules个最佳规则
            weight[cho] += len(batch[1])  # len(t_list)，每个batch的最佳规则，权重为，batch的t_list的长度，即tail的个数

        nonzero = (weight > 0)
        rules = self.rules_gen[nonzero]
        weight = weight[nonzero]

        if generator_version >= 2:
            rules = rules[:, 1:]
            return self.r, rules, weight

        inp = rules[:, :-1]
        tar = rules[:, 1:]
        gen_pad = self.R
        mask = (tar != gen_pad)
        del self.tmp__rule_embed
        return inp, tar, mask, weight

    def train_model(self):
        # self.make_batchs()
        train_batch = self.train_batch
        valid_batch = self.valid_batch
        test_batch = self.test_batch
        model = self
        print = self.print
        batch_size = self.arg('predictor_batch_size')
        num_epoch = self.arg('predictor_num_epoch')  # / batch_size
        lr = self.arg('predictor_lr')  # * batch_size
        print_epoch = self.arg('predictor_print_epoch')
        valid_epoch = self.arg('predictor_valid_epoch')

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 5)

        self.best = Metrics.init_value()
        self.best_model = self.state_dict()

        def train_step(batch):
            self.train()
            loss,loss_rule, _ = self(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            return loss,loss_rule

        def metrics_score(result):
            result = Metrics.pretty(result)
            mr = result['mr']
            mrr = result['mrr']
            h1 = result['h1']
            h3 = result['h3']
            h10 = result['h10']

            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            return self.arg('metrics_score_def', apply=eval_ctx(locals()))

        def format(result):
            s = ""
            for k, v in Metrics.pretty(result).items():
                if k == 'num':
                    continue
                s += k + ":"
                s += "%.4lf " % v
            return s

        def valid():#验证时，切换最好的模型步骤
            result = self._evaluate(valid_batch)
            updated = False
            if metrics_score(result) > metrics_score(self.best):
                updated = True
                self.best = result
                self.best_model = copy.deepcopy(self.state_dict())
            print(f"valid = {format(result)} {'updated' if updated else ''}")
            return updated, result

        last_update = 0
        cum_loss = 0
        valid()

        relation_embed_init = self.rotate.relation_embed.clone()
        ret_loss = 0
        ret_loss_rule =  0.0
        if len(train_batch) == 0:
            num_epoch = 0
        if self.em == 0:#TODO 第一epoch，不进行训练。
            num_epoch = 0
        print(f"Candidate rules:{self.rules_exp[:10]}/{len(self.rules_exp)}rules.")
        for epoch in range(1, num_epoch + 1):
            if epoch % max(1, len(train_batch) // batch_size) == 0:
                from random import shuffle
                shuffle(train_batch)
            batch = [train_batch[(epoch * batch_size + i) % len(train_batch)] for i in range(batch_size)]
            #print("len(batch)",len(batch))
            if epoch % print_epoch == 0:self.train_print = True
            loss,loss_rule = train_step(batch)
            cum_loss += loss.item()
            ret_loss += loss.item()
            ret_loss_rule = ret_loss_rule+loss_rule
            self.train_print = False
            if epoch % print_epoch == 0:#每print_epoch个步骤打印一次train_predictor训练信息
                print(epoch,len(batch),"tail:",self.t_list)
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_predictor #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                #print(f"rule loss = {loss_rule[:10]},ret_loss_rule = {ret_loss_rule[:10]}")
                print(f"ret_loss_rule = {ret_loss_rule[:10]}")
                cum_loss *= 0

            if epoch % valid_epoch == 0:#每valid_epoch个步骤打印一次predictor的valid信息
                if valid()[0]:
                    last_update = epoch
                elif epoch - last_update >= self.arg('predictor_early_break_rate') * num_epoch:
                    print(f"Early break: Never updated since {last_update}")
                    break
                if 1 - 1e-6 < Metrics.pretty(self.best)['mr'] < 1 + 1e-6:
                    print(f"Early break: Perfect")
                    break
        if num_epoch>0:
            ret_loss = ret_loss/num_epoch#scalar
            ret_loss_rule = ret_loss_rule/num_epoch
        else:
            ret_loss = 0

        with torch.no_grad():#采用best_model修正rotate的表征
            self.load_state_dict(self.best_model)
            self.rotate.relation_embed *= 0
            self.rotate.relation_embed += relation_embed_init
            self.rule_weight_raw[0] += 1000.0
            valid()

        self.load_state_dict(self.best_model)
        best = self.best
        if self.arg('record_test'):
            backup = self.recording
            self.record = []
            self.recording = True
        test = self._evaluate(test_batch)
        if self.arg('record_test'):
            self.recording = backup
        #best[0]存放取得best结果时的样本序号。
        print("__V__\t" + ("\t".join([str(self.r), str(int(best[0]))] + list(map(lambda x: "%.4lf" % x, best[1:])))))
        print("__T__\t" + ("\t".join([str(self.r), str(int(test[0]))] + list(map(lambda x: "%.4lf" % x, test[1:])))))
        #test[0]为该轮中所有的t_list之和，后面依次为mrr,mr,h1,h3,h10
        if self.em>0:
            self.result_sum.append(test)
        return best, test,ret_loss,ret_loss_rule


class CbGATGenerator(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, hidden_dim,Rh,Rt, print=print):
        super(CbGATGenerator, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mov = num_relations // 2
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = num_relations
        self.padding_idx = self.num_relations + 1
        self.num_layers = 1
        self.use_cuda = True
        self.r = -1
        self.Rh = Rh#(r,h_set())
        self.Rt = Rt#(r,t_set())
        #self.session_groundings = groundings
        self.rules_path = []
        self.path_gnd = []#路径gndlist，形式为[[{h:count,...},{gnd:count,...},...{t:count,...}],...]
        #self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.prec = []
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.print = print

        self.cuda()

    def inv(self, r):
        if r < self.mov:
            return r + self.mov
        else:
            return r - self.mov

    def zero_state(self, batch_size):
        state_shape = (self.num_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def forward(self, inputs, relation, hidden):
        pass
    #inputs：目标谓词，规则体（batchsize个定长一维list，长度为rulelenth+1，填充padding_index=num_relations + 1）；
	# target：样本里的规则体，终止id：ending_idx（shape为（batchsize,rulelenth+1)的tensor，填充padding_index=num_relations + 1）;
	# weight：Hscore;
    def loss(self, inputs, target, mask, weight):
        return 0
        
    
    #根据评估器返回的权重，训练一个由目标谓词生成规则序列的RNN
    #ret_loss_rule.shape = (num_rules,1);self.rules_path是长度num_rules的list，每个元素是一个规则path组成的tuple，可转为二维list
    #self.path_gnd：路径gndlist，形式为[[{h:count,...},{gnd:count,...},...{t:count,...}],...]
    def train_model(self,r, ret_loss,ret_loss_rule,em_epoch = 1,num_epoch=1, num_em_epoch=100):
        print = self.print
        #opt = torch.optim.Adam(self.parameters(), lr=lr)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 10)
        if CUDA:
            self.model_gat.cuda()
            self.cb_model_gat.cuda()
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'gpus')
                model_gat = nn.DataParallel(self.model_gat, device_ids=[torch.cuda.current_device()])
                cb_model_gat = nn.DataParallel(self.cb_model_gat, device_ids=[torch.cuda.current_device()])
            else:
                model_gat = self.model_gat
                cb_model_gat = self.cb_model_gat
        #optimizer = torch.optim.Adam(self.model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
        #cb_optimizer = torch.optim.Adam(self.cb_model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
        indiv_params =  []
        idx = 0
        for p in cb_model_gat.parameters():
            if idx!=2:indiv_params += [p]
            idx += 1     
        opt = torch.optim.Adam([{'params': model_gat.parameters()},
                                {'params': indiv_params}],
                                lr=10*args.lr, weight_decay=args.weight_decay_gat)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5, last_epoch=-1)

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
        #ret_loss_rule.shape = (num_rules,1);self.rules_path是长度num_rules的list，每个元素是一个规则path组成的tuple，可转为二维list
        #self.path_gnd：路径gndlist，形式为[[{h:count,...},{gnd:count,...},...{t:count,...}],...]    
        def rule_loss(ret_loss_rule,rule_path,path_gnd,metric_final = True):
            num_rule = len(rule_path)
            if metric_final:#TODO 是否采用final embedding
                metric_emb = self.r_emb
            else:
                metric_emb = self.model_gat.relation_embeddings
            #loss = torch.zeros(num_rule).cuda().requires_grad_()
            path_loss = []
            target_r = self.r
            #Cut wrong path;TODO C++ implementation,add(h,t) to rule sample
            #t_list = self.Rt[target_r]
            path_emb_list = []
            #sret_loss_rule = ret_loss_rule.softmax(dim=-1)#TODO
            print(f"ret_loss_rule:{ret_loss_rule[:10]}")
            for i,path in enumerate(rule_path):
                if i<1:
                    #loss[i]=0
                    continue
                rule = list(path)
                eval_loss = ret_loss_rule[i].item()/100+1e-5
                gnd_list = path_gnd[i][1:-1]#TODO,add cb_r:h_list,t_list
                print(f"Rule {rule} evaluator loss {eval_loss}.Length=",len(gnd_list),len(rule))
                assert len(gnd_list)==len(rule)-1
                path_emb = 0
                last_r = -1
                for j,r in enumerate(rule):
                    if last_r>=0:
                        if last_r not in self.r_dist.keys():
                            print("Error last_r",last_r)
                        elif self.r_dist[last_r] is None:
                            print("Error last_r",last_r)
                        else:
                            if self.r_dist[last_r][r] is not None:
                                path_emb = path_emb + self.r_dist_path(last_r,r,gnd_list[j-1],metric_final)
                            else:
                                print("Error last_r&r",last_r,r)
                    if r < self.R:
                        path_emb = path_emb + metric_emb[r]
                    else:
                        path_emb = path_emb - metric_emb[r]
                    last_r = r
                #path_emb_list.append(path_emb)
                path_loss.append((cos(path_emb,metric_emb[target_r])/eval_loss).unsqueeze(-1)) 
            # path_emb_all = torch.stack(path_emb_list)#(num_rule,emb_size)
            # target_emb = metric_emb[target_r].unsqueeze(0).repeat(self.num_rule,1)
            # y = torch.ones(len(num_rule)).cuda()
            # loss = gat_loss_func(path_emb_all, target_emb, y)
            #loss_norm = torch.norm(loss, p=1, dim=1)
            #print(path_loss[0])
            loss = torch.cat(path_loss)
            return loss.mean()


        
        cum_loss = 0
        #em_loss = torch.tensor(0.0).cuda().requires_grad_() + ret_loss
        epoch_losses = []
        init_flag = True
        em_weight = 0.5
        #num_epoch = 1 #gen的epoch
        for epoch in range(1, num_epoch + 1):
            print("\nepoch-> ", epoch)
            random.shuffle(Corpus_.train_triples)
            random.shuffle(cb_Corpus_.train_triples)
            Corpus_.train_indices = np.array(
                list(Corpus_.train_triples)).astype(np.int32) ##转为int格式的训练三元组id数据
            cb_Corpus_.train_indices = np.array(
                list(cb_Corpus_.train_triples)).astype(np.int32)

            model_gat.train()  # getting in training mode，这里仅是切换为train模式
            cb_model_gat.train()
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
            print("\n len(cb_Corpus_.train_indices) ==", len(cb_Corpus_.train_indices))

            for iters in range(num_iters_per_epoch):
                print("\n iters-> ", iters)
                start_time_iter = time.time()
                train_indices, train_values = Corpus_.get_iteration_batch(iters)#循环取出第iter_num个batch,同时生成正负样本，label分别是1、-1
                cb_train_indices, cb_train_values = cb_Corpus_.get_iteration_batch(iters) #数量应该不一致，主要是负样本数量不同
                print("\n len(train_indices) ==", len(train_indices))
                print("\n len(cb_train_indices) ==", len(cb_train_indices))
                if CUDA:
                    train_indices = Variable(
                        torch.LongTensor(train_indices)).cuda()
                    train_values = Variable(torch.FloatTensor(train_values)).cuda()
                    cb_train_indices = Variable(
                        torch.LongTensor(cb_train_indices)).cuda()
                    cb_train_values = Variable(torch.FloatTensor(cb_train_values)).cuda()

                else:
                    train_indices = Variable(torch.LongTensor(train_indices))
                    train_values = Variable(torch.FloatTensor(train_values))
                    cb_train_indices = Variable(torch.LongTensor(cb_train_indices))
                    cb_train_values = Variable(torch.FloatTensor(cb_train_values))
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
                
                #update emb
                if torch.cuda.device_count() > 1:
                    self.model_gat = model_gat.module
                    self.cb_model_gat = cb_model_gat.module
                else:
                    self.model_gat = model_gat
                    self.cb_model_gat = cb_model_gat
                #update final emb
                self.e_emb = self.model_gat.final_entity_embeddings
                self.r_emb = self.model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
                self.cb_e_emb  = self.cb_model_gat.final_entity_embeddings
                self.cb_r_emb = self.cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
           
                opt.zero_grad()
                # loss = batch_gat_loss(
                #     gat_loss_func, train_indices, entity_embed, relation_embed)
                # cb_loss = batch_gat_loss(
                #     gat_loss_func, cb_train_indices, cb_entity_embed, cb_relation_embed,mod = 1)
                rl_loss = rule_loss(ret_loss_rule,self.rules_path,self.path_gnd,metric_final = True)#torch.nn.CosineEmbeddingLoss(reduction = "mean")
                #loss = (1-em_weight)*(loss + cb_loss)+10*em_weight*rl_loss #+ em_weight*ret_loss
                loss = rl_loss
                loss.backward()
                opt.step()
                #sch.step()
                end_time_iter = time.time()
                epoch_loss.append(loss.data.item())

                #cum_loss += loss.item()
                print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
                
            
            sch.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            epoch_mean_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_losses.append(epoch_mean_loss)

            #if epoch % print_epoch == 0:
            #lr_str = "%.2e" % (opt.param_groups[0]['lr'])
            print(f"train_generator #{epoch} epoch_mean_loss = {epoch_mean_loss}")
        if em_epoch%10==0 or em_epoch == num_em_epoch - 1:
            save_model(model_gat, args.data, em_epoch,
                    args.output_folder,prex = 'em_e')
            save_model(cb_model_gat, args.data, em_epoch,
                   args.output_folder,prex = 'em_r')
            
        if torch.cuda.device_count() > 1:
            self.model_gat = model_gat.module
            self.cb_model_gat = cb_model_gat.module
        else:
            self.model_gat = model_gat
            self.cb_model_gat = cb_model_gat

    # gen rules from rule_file
    def rule_gen(self,r,rule_file,num_samples, max_len):

        rules = []
        rule_set = set([tuple(), (r,)])

        #更新r_dist
        self.e_emb = self.model_gat.final_entity_embeddings
        self.r_emb = self.model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
        self.cb_e_emb  = self.cb_model_gat.final_entity_embeddings
        self.cb_r_emb = self.cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
        #self.r_dist = self.patch_r_dist(r_num = 2*self.R)
        has_inv = False
        with open(rule_file) as file:
            for i, line in enumerate(file):
                #try:
                path, prec = line.split('\t')
                path = tuple(map(int, path.split()))
                prec = float(prec.split()[0])#采用rulesample的grpundingtruth评估结果prec作为初始的规则权重，意义为h出发的所有的rule path中，命中t的path的比例

                if not (prec >= 0.0001):
                    # to avoid negative and nan
                    prec = 0.0001
                if path in rule_set:
                    continue
                for rel in path:
                    if rel >= self.R:#237
                        has_inv = True
                        break
                if has_inv:
                    has_inv = False
                    #print("Jump inv_rel rule:",path)
                    continue#去掉rulefile里的逆关系规则
                #print("path",path)
                #print("Calc score for rule:",path)
                gnd_list = []
                sc ,gnd_list = self.cb_rule_score_gnd(r,path)
                print(path,"Score  = ",sc,"sample prec",prec)
                rule_set.add(path)
                if len(path) <= max_len:
                    rules.append((path,gnd_list, sc,i,prec))
                    #self.path_gnd.append((gnd_list, sc,i))
                # except:
                #     continue
        #print("____len(rules),num_samples____",len(rules),len(rule_set),num_samples) 
        #print("____rules____",rules)            
        rules = sorted(rules, key=lambda x: (x[2], x[3]), reverse=True)[:num_samples]#利用权重排序rule
        rules = [((r,),[dict()], 1, -1,1)]+rules#TODO,r的h_list/t_list
        #rule_set = set([tuple(), (r,)])

        print(f"CbGAT generate candidate rules: |rules| = {len(rules)} max_rule_len = {max_len}")
        x = torch.tensor([sc for _,_, sc,_,_ in rules]).cuda()
        #prior = -torch.log((1 - x.float()).clamp(min=1e-6))#最好是0-1
        prior = x
        self.path_gnd = [gnd_list for _,gnd_list, _, _,_ in rules]#同步更新生成的各规则的路径gndlist，形式为[[{h:count,...},{gnd:count,...},...{t:count,...}],...]
        self.prec = [prec for _,_, _, _,prec in rules]
        rules = [path for path, _,_, _,_ in rules]        
        return rules,prior
    #undireted,等增加CBGAT的有向向量学习
    def cb_rule_score(self,target_r,path):
        
        rule_path = list(path)
        path_emb = torch.zeros(self.r_emb_len).cuda()
        last_r = -1
        for r in rule_path:
            if last_r>=0:
                if last_r not in self.r_dist.keys():
                    return 1e-8
                elif self.r_dist[last_r] is None:
                    return 1e-8
                else:
                    if self.r_dist[last_r][r] is not None:
                        path_emb = path_emb + self.r_dist[last_r][r]
                    else:
                        return 1e-8
            if r < self.R:
                path_emb = path_emb + self.r_emb[r]
            else:
                path_emb = path_emb - self.r_emb[r]
            last_r = r
        output = cos(path_emb,self.r_emb[target_r])#(1,-1)
        #angs = (torch.acos(output)*180/3.1415926).item()
        angs = output.item()
        #print("Score for rule",rule_path," = ",angs)
        return angs
    
    def cb_rule_score_gnd(self,target_r,path):
        gnd_list = []#规则路径上所有的gnd实体及其路径次数count
        h_list = list(self.Rh[target_r])#TODO
        
        hgnd = dict()
        for h in h_list:
            hgnd[h]=1
        if len(hgnd)>0:
            gnd_list.append(hgnd)
        else:
            gnd_list.append(None)

        rule_path = list(path)
        path_emb = torch.zeros(self.r_emb_len).cuda()
        last_r = -1
        cur_path = []
        gnd = None
        for r in rule_path:
            if last_r>=0:
                if last_r not in self.r_dist.keys():
                    gnd_list.append(None)
                    return 1e-8
                elif self.r_dist[last_r] is None:
                    gnd_list.append(None)
                    return 1e-8
                else:
                    if self.r_dist[last_r][r] is not None:
                        gnd = self.h_path_t(h_list,cur_path)
                        gnd_list.append(gnd)
                        path_emb = path_emb + self.r_dist_path(last_r,r,gnd)
                    else:
                        gnd_list.append(None)
                        return 1e-8
            if r < self.R:#取读取模型的关系表征的长度
                path_emb = path_emb + self.r_emb[r]
            else:
                path_emb = path_emb - self.r_emb[r-self.R]
            last_r = r
            cur_path.append(r)
        #if gnd is not None:
        tgnd = self.h_path_t(h_list,cur_path)
        if len(tgnd)>0:
            gnd_list.append(tgnd)
        else:
            gnd_list.append(None)

        output = cos(path_emb,self.r_emb[target_r])#(1,-1)
        #angs = (torch.acos(output)*180/3.1415926).item()
        angs = output.item()
        #self.path_gnd.append(gnd_list)
        print("Score for rule",rule_path," = ",angs)
        return angs,gnd_list

    def h_path_t(self,h_list,path):#返回{t:count}
        gnd = dict()
        for h in h_list:
            hgnd = calc_groundings(h, path,count=True)
            for key,value in hgnd.items():
                if key in gnd:
                    gnd[key] += value
                else:
                    gnd[key] = value
        return gnd
    def r_dist_path(self,last_r,r,gnd,metric_final = True): #gnd中的t即为last_r,r之间的cb_relation
        # e_list = self.cb_kg[last_r][r]
        # for t in gnd.keys():
        #     if t not in e_list:
        #         print("Error gnd for last_rel & rel",last_r,r)
        if metric_final:
            metric_emb = self.cb_r_emb
        else:
            metric_emb = self.cb_model_gat.relation_embeddings
        
        mide =  self.cb_kg[last_r][r]
        #print(mid_e.shape)
        #cb_r_list = list(set(mide)&set(gnd.keys()))
        link = dict()
        for k,v in gnd.items():
            if k in mide:
                link[k]=v
        #print("len(self.cb_kg[last_r][r]),len(gnd),len(cb_r_list):",len(mide),len(gnd),len(link))
        wt = torch.tensor(list(link.values())).float()
        wt_norm = torch.nn.functional.normalize(wt,p=1,dim=0).cuda() 
        #mid_e_emd = metric_emb[list(gnd.keys()),:].cuda()
        mid_e_emd = metric_emb[list(link.keys()),:].cuda()
        mean_mid_e_emd = mid_e_emd*wt_norm.unsqueeze(-1)
        r_dist = mean_mid_e_emd.sum(dim=0)
        #print(mid_e_emd.shape)
        return r_dist

    def cb_rule_score_directed(self,target_r,path):
        rule_path = list(path)
        path_emb = torch.zeros(self.r_emb_len).cuda()
        last_r = -1
        for r in rule_path:
            if last_r>=0 and self.r_dist[last_r][r] is not None:
                path_emb = path_emb + self.r_dist[last_r][r]
            elif last_r>=0:
                if last_r>=self.R and r>=self.R:
                    path_emb = path_emb - self.r_dist[r-self.R][last_r-self.R]

            if r < self.R:#取读取模型的关系表征的长度
                path_emb = path_emb + self.r_emb[r]
            else:
                path_emb = path_emb - self.r_emb[r-self.R]
            last_r = r
        output = cos(path_emb,self.r_emb[target_r])
        #angs = (torch.acos(output)*180/3.1415926).item()
        angs = output.item()
        print("Score for rule",rule_path," = ",angs)
        return angs

    def init(self):
        model_gat,cb_model_gat = read_cd_gat(args)
        self.model_gat = model_gat
        self.cb_model_gat = cb_model_gat
        
        self.e_emb = self.model_gat.final_entity_embeddings
        self.r_emb = self.model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
        self.cb_e_emb  = self.cb_model_gat.final_entity_embeddings
        self.cb_r_emb = self.cb_model_gat.final_relation_embeddings#得到GAT模型的最终嵌入
        self.e_emb_len = self.e_emb.shape[-1]
        self.r_emb_len = self.r_emb.shape[-1]
        self.E = self.e_emb.shape[0]
        self.R = self.r_emb.shape[0]
        # self.path =  self.model_gat.final_out_entity_l_1
        # self.cb_path = self.cb_model_gat.final_out_entity_l_1
        # print("self.path.shape,self.cb_path.shape:",self.path.shape,self.cb_path.shape)
        # self.att = self.model_gat.cubic_attention
        # self.cb_att = self.cb_model_gat.cubic_attention
        # print("self.att.shape,self.cb_att.shape:",self.att.shape,self.cb_att.shape)
        file = args.data + "/cb_Corpus_graph.pickle"
        if not os.path.exists(file):
            self.cb_kg = cb_Corpus_.get_multiroute_graph()
            file = args.data + "/cb_Corpus_graph.pickle"
            with open(file, 'wb') as handle:
                pickle.dump(self.cb_kg, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            self.kg = Corpus_.get_multiroute_graph()
            file = args.data + "/Corpus_graph.pickle"
            with open(file, 'wb') as handle:
                pickle.dump(self.kg, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading Generated graph  >>>")
            self.cb_kg = pickle.load(open(args.data + "/cb_Corpus_graph.pickle",'rb'))
            self.kg = pickle.load(open(args.data + "/Corpus_graph.pickle",'rb'))

        file = args.data + "/cb_path_graph.pickle"
        if not os.path.exists(file):
            self.cb_path_kg = cb_Corpus_.get_path_graph()
            file = args.data + "/cb_path_graph.pickle"
            with open(file, 'wb') as handle:
                pickle.dump(self.cb_path_kg, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            self.path_kg = Corpus_.get_path_graph()
            file = args.data + "/path_graph.pickle"
            with open(file, 'wb') as handle:
                pickle.dump(self.path_kg, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading Generated path_graph  >>>")
            self.cb_path_kg = pickle.load(open(args.data + "/cb_path_graph.pickle",'rb'))
            self.path_kg = pickle.load(open(args.data + "/path_graph.pickle",'rb'))

        file = args.data + "/r_dist_undirected.pickle"
        if not os.path.exists(file):
            self.r_dist = self.patch_r_dist(r_num = 2*self.R)
            with open(file, 'wb') as handle:
                pickle.dump(self.r_dist, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading Generated r_dist  >>>")
            self.r_dist = pickle.load(open(file,'rb'))

    #返回谓词r和邻接谓词的距离/生成器和评估器关于反关系一致时
    # cb_r_emd:cb_r的表征；cb_kg：cb_r的链接拓扑graph
    def mean_dist(self,r_num = None):
        print("Generating r_dist  >>>")
        if r_num == None:r_num = self.R
        dist_r = {}
        for rh in range(r_num):
            if rh not in self.cb_kg:
                print(rh,"is not a head predicate.")
                dist_r[rh] = None
                continue
            if len(self.cb_kg[rh])==0:
                dist_r[rh] = None
                continue
            dist_r[rh] = {}
            for next_r in range(r_num):
                if next_r in self.cb_kg[rh].keys():
                    mid_e = torch.Tensor(self.cb_kg[rh][next_r]).long()#rh和next_r之间所有cb_r（实体）的id
                    #print(mid_e.shape)
                    mid_e_emd = self.cb_r_emb[mid_e,:]
                    #print(mid_e_emd.shape)
                    mean_mid_e_emd = mid_e_emd.mean(dim = 0)
                    dist_r[rh][next_r] = mean_mid_e_emd
                else:
                    dist_r[rh][next_r] = None
        #for rh in range(r_num,):
        return dist_r

    #修正生成器无反关系的情况,全反关系取负
    def patch_r_dist(self,r_num = None):
        if r_num <= self.R:
            self.r_dist = self.mean_dist(r_num)
            return self.r_dist
        else:
            self.r_dist = self.mean_dist(r_num - self.R)
        

        for rh in range(r_num):
            inv_rh = rh - self.R
            if rh not in self.r_dist.keys():
                self.r_dist[rh]=dict()
            for next_r in range(r_num): 
                inv_next_r = next_r - self.R
                if inv_rh<0 and inv_next_r <0 :continue
                if inv_rh>=0 and inv_next_r >=0:
                    if inv_next_r in self.r_dist.keys():
                        if self.r_dist[inv_next_r] is not None:
                            if inv_rh in self.r_dist[inv_next_r].keys():                            
                                if rh not in self.r_dist.keys():self.r_dist[rh] = dict()
                                if self.r_dist[inv_next_r][inv_rh] is not None: 
                                    self.r_dist[rh][next_r]= - self.r_dist[inv_next_r][inv_rh]
                                    continue
                if rh not in self.r_dist.keys():
                    self.r_dist[rh] = None
                elif self.r_dist[rh] is not None:            
                    self.r_dist[rh][next_r]=None
        return self.r_dist



def read_cd_gat(args):
    print("Load model")
    #epoch_load = args.epochs_gat #默认取模型最后一个epcoh
    epoch_load = 0#默认初始化训练时
    epoch_load = 3000#载入之前的模型时。手动指定载入的模型epoch,注意是文件名里的数字+1

    #print("train_gat_cb entity_embeddings",entity_embeddings)
    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT

    #print("train_gat_cb entity_embeddings.shape,relation_embeddings.shape",entity_embeddings.shape,relation_embeddings.shape)
    #model_gat.to(out_device)
    #print("init final_entity_embeddings...",model_gat.final_entity_embeddings)

    pre  = 'cb_e'
    if epoch_load>0:
        model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_e{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)
    #print(model_gat.state_dict())

    #model_gat = SpKBGATModified(final_entity_embeddings, final_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    #                            args.drop_GAT, args.alpha, args.nheads_GAT) #GAT
    
    #print("train_gat_cb cb_relation_embeddings",cb_relation_embeddings)
    cb_model_gat = SpKBGATModified(model_gat.relation_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    args.drop_GAT, args.alpha, args.nheads_GAT,cb_flag=True) #cubic GAT

    #print("init cb_final_relation_embeddings...",cb_model_gat.final_relation_embeddings)


    pre  = 'cb_r'
    if epoch_load>0:
        cb_model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_r{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)

    #print("train_gat_cb final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape",final_entity_embeddings.shape,final_relation_embeddings.shape,cb_relation_embeddings.shape)
    
    if CUDA:
        model_gat.cuda()
        cb_model_gat.cuda()
        # if torch.cuda.device_count() > 1:
        #     print("Use", torch.cuda.device_count(), 'gpus')
        #     model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
        #     cb_model_gat = nn.DataParallel(cb_model_gat, device_ids=[torch.cuda.current_device()])

    return model_gat,cb_model_gat
