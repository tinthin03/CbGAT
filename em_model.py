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
from knowledge_graph_utils import *


#Each time train_Model is called, a target predicate needs to be specified.


if exp == 'fb15':
    if model_type == "rotatE":
        ORG_DATA_DIR = "./data/FB15k-237-rotate/"
    else:
        ORG_DATA_DIR = "./data/FB15k-237/"
elif exp == 'wn18':
    if model_type == "rotatE":
        ORG_DATA_DIR = "./data/WN18RR-rotate/"
    else:
        ORG_DATA_DIR = "./data/WN18RR/"

elif exp == 'umls':
    if model_type == "rotatE":
        ORG_DATA_DIR = "./data/umls-rotate/"
    else:
        ORG_DATA_DIR = "./data/umls/"
elif exp == 'ilpc':
    if model_type == "rotatE":
        ORG_DATA_DIR = "./data/ilpc-small-rotate/"
    else:
        ORG_DATA_DIR = "./data/ilpc-small/"
elif exp == 'ilpc-large':
    if model_type == "rotatE":
        ORG_DATA_DIR = "./data/ilpc-large-rotate/"
    else:
        ORG_DATA_DIR = "./data/ilpc-large/"


recall = True#False#True
# inductive = False
sim_model = 'cos'#cos#F1#F2
valuate_modelling = 0  #0:'one',1:'weight',2:'rotatE',3:'transE'，one：
path_model = 1 #  1:rel+cbrel;  2:rel+head+cbrel;  3:ln*headrel+(ln-j)*cbrel;  4:ln*headrel+cbrel;  5:
#model_type = 'rotatE'
max_rule_length = 4
rule_loss_aug = 1000#1000
max_rules_num = 290
max_beam_rules = 3000
Auto_best = 1 #0:Final models，1:self-adaption according to valid，2:test best



def calc_groundings(h, rule,r_groundings,count=False): # buffer when train graph
    # if inductive:
    #     return groundings.groundings(h, rule,count)
    path = ""
    for r in rule:
        path = path+str(r)+' '
    path = path[:-1]
    if path in r_groundings[h].keys():
        if not count:
            rgnd = list(r_groundings[h][path].keys())
        else:
            rgnd = r_groundings[h][path]
    else:
        rgnd = groundings.groundings(h, rule,count)
        r_groundings[h][path] = dict()
        if not count:
            for t in rgnd:
                if t not in r_groundings[h][path].keys():
                    r_groundings[h][path].update({t:1})
        else:
            #for t,cnt in rgnd
            r_groundings[h][path].update(rgnd)
    return rgnd

def h_path_t(h_list,path,r_groundings):#return {t:count}
        gnd = dict()
        for h in h_list:
            hgnd = calc_groundings(h, path,r_groundings,count=True)
            for key,value in hgnd.items():
                if key in gnd:
                    gnd[key] += value
                else:
                    gnd[key] = value
        return gnd


class EMiner(torch.nn.Module):
    def __init__(self, dataset, args, print=print):
        super(EMiner, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R']
        assert self.R % 2 == 0
        args = self.set_args(args)
        self.result = []
        self.rule_quality = []

        self.dataset = dataset
        self._print = print
        self.print = self.log_print

        #For inductive test
        if inductive:
            self.org_dataset = load_dataset(f"{ORG_DATA_DIR}",exp = exp)
        else:
            self.org_dataset = None

        self.predictor_init = lambda: Evaluator(self.dataset,self._args, print=self.log_print,org_dataset = self.org_dataset)
        self.generator = CbGATGenerator(self.R, self.arg('generator_embed_dim'), self.arg('generator_hidden_dim'),self.dataset['Rh'],self.dataset['Rt'],
                                           print=self.log_print)
        self.r_groundings = groundings.init_groundings()
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
        self.generator.r_groundings = self.r_groundings
        #self.predictor.r_groundings = self.r_groundings #X
        
        max_beam_rules = 3000
        def generate_rules():
            if self.em == 0:
                print("Use rule file to init.")
                self.predictor.relation_init(r=r, rule_file=rule_file, force_init_weight=self.arg('init_weight_boot'))
            else:
                sampled = set()
                sampled.add((r,))
                sampled.add(tuple())

                rules = [(r,)]#r
                prior = [0.0, ]
                grules, gscores = self.generator.rule_gen(r,rule_file,self.arg('max_beam_rules'),self.predictor.arg('max_rule_len'),modelling=self.predictor.arg('modelling'))
                i = 0
                toprule_quality = 0
                allrule_quality = 0
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
                    if len(sampled) < 20:
                        toprule_quality += self.generator.prec[i]
                    allrule_quality += self.generator.prec[i]
                    i += 1
                avg_quality = allrule_quality/len(grules)
                print(f"Done |sampled| = {len(sampled)},Top 20 rules quality = {toprule_quality},all = {allrule_quality},avg = {avg_quality}")
                self.rule_quality.append([toprule_quality,allrule_quality,avg_quality])
                prior = torch.tensor(prior).cuda()
                # prior -= prior.max()
                # prior = prior.exp()
                
                self.predictor.relation_init(r, rules=rules,path_gnd = self.generator.path_gnd,prec = self.generator.prec,rule_emb = self.generator.rule_emb, prior=prior)
                
                self.generator.path_gnd = self.predictor.path_gnd
                self.generator.prec = self.predictor.prec
                if self.predictor.arg('modelling')==3:
                    self.generator.rule_emb = self.predictor.rule_emb
                self.generator.rules_path = self.predictor.rules_exp

        for self.em in range(num_em_epoch):
            if self.em > 0:
                pem = self.predictor.em
                presult_sum = self.predictor.result_sum
                pt_list = self.predictor.t_list
                ptrain_print = self.predictor.train_print
                bestMRR = self.predictor.bestMRR
                self.predictor = self.predictor_init()
                self.predictor.em = pem
                self.predictor.result_sum = presult_sum#[[0.0,0,0,0,0]]
                self.predictor.t_list = pt_list#temp store t_list in train_step
                self.predictor.train_print = ptrain_print
                self.predictor.bestMRR = bestMRR
                self.predictor.r_groundings = self.r_groundings
            else:
                self.predictor = self.predictor_init()
                self.predictor.r_groundings = self.r_groundings
            self.predictor.pgnd_buffer = pgnd_buffer
            self.predictor.rgnd_buffer = rgnd_buffer
            self.predictor.rgnd_buffer_test = rgnd_buffer_test
            self.predictor.em = self.em

            self.generator.r = self.r
            
            
            generate_rules()
            if self.predictor.arg('modelling')==3:
                self.predictor.e_emb = self.generator.e_emb
            # E-Step:
            print("Train/test Evaluator(E_step).")
            valid, test,ret_loss,ret_loss_rule = self.predictor.train_model()#执行规则评估器的训练，设置了self.training为真

            # M-Step
            if self.em>0:#TODO
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
            #torch.save(ckpt, model_file)
            gc.collect()

        # Testing
        self.em = num_em_epoch
        self.predictor.em = self.em
        generate_rules()
        valid, test,ret_loss,ret_loss_rule = self.predictor.train_model()
        self.result = self.predictor.result_sum[-1]#last and lastlast compare
        self.result_quality = torch.cat([torch.Tensor([int(self.result[0])]),torch.Tensor(self.rule_quality).mean(dim=0)])
        self.rule_quality = []

        self.r_groundings.clear()
        self.r_groundings = groundings.init_groundings()
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
        def_args['num_em_epoch'] = 9
        def_args['sample_print_epoch'] = 20
        def_args['max_beam_rules'] = 3000#Total number of rules generated by the generator or filtered by rulefile.
        def_args['generator_embed_dim'] = 512
        def_args['generator_hidden_dim'] = 256
        def_args['generator_lr'] = 1e-3
        def_args['generator_num_epoch'] = 10000
        def_args['generator_print_epoch'] = 100
        def_args['init_weight_boot'] = False
        #def_args['modelling'] = 3

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
    def __init__(self, dataset, args, print=print,org_dataset = None):
        super(Evaluator, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R'] + 1 #237*2+1
        assert self.R % 2 == 1
        self.dataset = dataset
        self.org_dataset = org_dataset


        self.set_args(args)
        rotate_pretrained = self.arg('rotate_pretrained', apply=lambda x: x)
        self.rotate = RotatE(dataset, rotate_pretrained)
        self.training = True
        self.e_emb = None
        self.r_groundings = None

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
        #if firstinit:
        #print("Evaluator firstinit.........")
        self.em = 0
        self.result_sum = []#[[0.0,0,0,0,0]]
        self.t_list = []#temp store t_list in train_step
        self.train_print = False
        #self.modelling = self.arg('modelling')
        self.bestMRR = -1

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

    def rule_embed(self, force=False):
        if not force and not self.arg('param_relation_embed'):
            return self._rule_embed

        relation_embed = self.rotate._attatch_empty_relation()
        rule_embed = torch.zeros(self.num_rule, self.rotate.embed_dim).cuda()
        for i in range(self.MAX_RULE_LEN):
            rule_embed += self.index_select(relation_embed, self.rules[i])
        return rule_embed#shape = (self.num_rule, self.rotate.embed_dim)


    def set_rules(self, rules):
        paths = rules
        r = self.r
        self.eval()

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
    def pgnd(self, h, i, num=None, rgnd=None):
        if num is None:
            num = self.arg('pgnd_num')

        key = (h, self.r, tuple(self.rules_exp[i]))
        if key in self.pgnd_buffer:
            return self.pgnd_buffer[key]

        with torch.no_grad():
            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()
            rule_embed = self.rotate.embed(h, self.tmp__rule_embed[i])#hOr
            if ans_can is None:
                dist = self.rotate.dist(rule_embed, self.rotate.entity_embed)#
            else:
                can_dist = self.rotate.dist(rule_embed, self.rotate.entity_embed[ans_can])
                dist = torch.zeros(self.E).cuda() + 1e10
                dist[ans_can] = can_dist

            if rgnd is not None:
                dist[torch.LongTensor(rgnd).cuda()] = 1e10#
            ret = torch.arange(self.E).cuda()[dist <= self.rotate.gamma]

            dist[ret] = 1e10
            num = min(num, dist.size(0) - len(rgnd)) - ret.size(-1)
            if num > 0:
                tmp = dist.topk(num, dim=0, largest=False, sorted=False)[1]
                ret = torch.cat([ret, tmp], dim=0)

        self.pgnd_buffer[key] = ret
        ##########
        # print(h, sorted(ret.cpu().numpy().tolist()))
        return ret


    def cscore(self, rule_embed, crule, centity, cweight):
        score = self.rotate.compare(rule_embed, self.rotate.entity_embed, crule, centity)
        score = (self.rotate.gamma - score).sigmoid()
        if self.arg('drop_neg_gnd'):
            score = score * (score >= 0.5)
        score = score * cweight
        return score

    def rule_value(self, batch, weighted=False):
        num_rule = self.num_rule
        h, t_list, mask, crule, centity, cweight = self.load_batch(batch)
        # print("rule_value--->h,t_list",h,t_list)
        with torch.no_grad():

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed)#hOt
            cscore = self.cscore(rule_embed, crule, centity, cweight)
            indices = torch.stack([crule, centity], 0)

            def cvalue(cscore):
                if cscore.size(0) == 0:
                    return torch.zeros(num_rule).cuda()
                return torch.sparse.sum(torch.sparse.FloatTensor(
                    indices,
                    cscore,
                    torch.Size([num_rule, self.E])
                ).cuda(), -1).to_dense()


            pos = cvalue(cscore * mask[centity])
            neg = cvalue(cscore * ~mask[centity])
            score = cvalue(cscore)
            num = cvalue(cweight).clamp(min=0.001)

            pos_num = cvalue(cweight * mask[centity]).clamp(min=0.001)
            neg_num = cvalue(cweight * ~mask[centity]).clamp(min=0.001)


            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)
            #'rule_value_def',  (pos - neg) / num
            value = self.arg('rule_value_def', apply=eval_ctx(locals()))
            
            if weighted:
                value *= len(t_list)

            if hasattr(self, 'tmp__rule_value'):
                self.tmp__rule_value += value#
                self.tmp__num_init += len(t_list)

        return value

    # Choose rules, which has top `num_samples` of `value` and has a non negative `nonneg`
    def choose_rules(self, value, nonneg=None, num_samples=None, return_mask=False):
        if num_samples is None:
            num_samples = self.arg('max_best_rules')
        ################
        # print(f"choose_rules num = {num_samples}")
        with torch.no_grad():
            num_rule = value.size(-1)
            topk = value.topk(min(num_samples - 1, num_rule), dim=0, largest=True, sorted=False)[1]
            cho = torch.zeros(num_rule).bool().cuda()
            cho[topk] = True
            if nonneg is not None:
                cho[nonneg < 0] = False

        if return_mask:
            return cho
        return mask2list(cho)

    # Choose best rules for each batch, for M-step
    def best_rules(self, batch, num_samples=None):
        with torch.no_grad():
            w = self.rule_value(batch)
            value = (w + self.arg('prior_coef') * self.prior) * self.rule_weight
            cho = self.choose_rules(value, nonneg=w, num_samples=num_samples)
        return cho

    # For a new relation, init rule weights and choose rules
    def relation_init(self, r=None, rule_file=None,path_gnd = [],prec = [],rule_emb = [], rules=None, prior=None, force_init_weight=False):
        print = self.print
        if r is not None:
            self.r = r
        r = self.r
        if rules is None:
            assert rule_file is not None
            rules = [((r,), 1, -1)]
            rule_set = set([tuple(), (r,)])
            has_inv = False
            with open(rule_file) as file:
                for i, line in enumerate(file):
                    try:
                        path, prec = line.split('\t')
                        path = tuple(map(int, path.split()))
                        prec_ = float(prec.split()[0])
                        recall_ = float(prec.split()[1])
                        
                        if prec_> 1 or prec_<0 :
                            # to avoid negative and nan
                            prec_ = 0.0001

                        if recall:
                            prec = prec_ * recall_* 100
                        else:
                            prec = prec_
                        if not (prec >= 0.0001):
                            # to avoid negative and nan
                            prec = 0.0001

                        if path in rule_set:
                            continue
                        
                        if not inv_relatation:
                            for rel in path:
                                if rel >= (self.R-1)/2 and rel != r:
                                    has_inv = True
                                    break
                            if has_inv:
                                has_inv = False
                                continue
                        rule_set.add(path)
                        if len(path) <= self.arg('max_rule_len'):
                            rules.append((path, prec, i))
                    except:
                        continue

            rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)[:self.arg('max_beam_rules')]
            print(f"Loaded from file: |rules| = {len(rules)} max_rule_len = {self.arg('max_rule_len')}")
            x = torch.tensor([prec for _, prec, _ in rules]).cuda()
            prior = -torch.log((1 - x.float()).clamp(min=1e-6))
            # prior = x
            rules = [path for path, _, _ in rules]
            if self.arg('modelling')==3:#transE
                for path in rules:            
                    path_emb = torch.zeros(1).cuda()
                    rule_emb.append(path_emb)

        else:
            assert prior is not None

        self.prior = prior
        self.set_rules(rules)
        self.path_gnd = path_gnd
        self.prec = prec
        print("Generator exctract max_beam_rules/num_rule.|rules|:",self.num_rule)
        num_rule = self.num_rule
        with torch.no_grad():
            self.tmp__rule_value = torch.zeros(num_rule).cuda()
            self.tmp__rule_embed = self.rule_embed(force=True).detach()
            self.tmp__num_init = 0
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        init_weight = force_init_weight or not self.arg('init_weight_with_prior')
        if self.arg('rely_gen'):
            init_weight = not self.arg('rely_gen') #False
        if init_weight:
            for batch in self.make_batchs(init=True):
                self.rule_value(batch, weighted=True)

        with torch.no_grad():
            avg_rule_value = self.tmp__rule_value / max(self.tmp__num_init, 1)
            print("Total tmp__rule_value,total avg(tmp__rule_value),self.prior",self.tmp__rule_value.sum(),avg_rule_value.sum(),self.prior.sum())
            print(self.prior.shape,self.prior[0:50])
            value =  avg_rule_value + self.arg('prior_coef') * self.prior
            nonneg = self.tmp__rule_value
            if self.arg('use_neg_rules') or not init_weight:
                nonneg = None
            cho = self.choose_rules(value, num_samples=self.arg('max_rules'), nonneg=nonneg, return_mask=True)
            print("choose max_rules,update rule_weight_raw for Evaluator training.")
            cho[0] = True
            cho_list = mask2list(cho).detach().cpu().numpy().tolist()
            value_list = value.detach().cpu().numpy().tolist()
            cho_list = sorted(cho_list,
                              key=lambda x: (x == 0, value_list[x]), reverse=True)
            assert cho_list[0] == 0
            cho = torch.LongTensor(cho_list).cuda()

            value = value[cho]
            self.tmp__rule_value = self.tmp__rule_value[cho]
            self.prior = self.prior[cho]
            self.rules = self.rules[:, cho]
            self.rules_gen = self.rules_gen[cho]
            self.rules_exp = [self.rules_exp[x] for x in cho_list]
            #print("len(self.path_gnd),len(cho_list)",len(self.path_gnd),len(cho_list))
            if len(self.path_gnd)>0:
                self.path_gnd = [self.path_gnd[x] for x in cho_list]
                self.prec  = [self.prec[x] for x in cho_list]
            if self.arg('modelling')==3:#transE
                self.rule_emb = [rule_emb[x] for x in cho_list]

        if init_weight:#
            weight = self.tmp__rule_value+self.prior*1000
        else:
            weight = self.prior

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

        self.make_batchs()
        print("Use max_rules,update pgnd for Evaluator training.")
        del self.tmp__rule_embed

    # Default arguments for predictor
    def set_args(self, args):
        self._args = dict()
        def_args = dict()
        def_args['rotate_pretrained'] = None
        def_args['max_beam_rules'] = max_beam_rules#Total number of rules generated by the generator or filtered by rulefile
        def_args['max_rules'] = max_rules_num#For example, if crule=[0,0,1., 1., 1., 2., 2., 2.], then 3。
        def_args['max_rule_len'] = max_rule_length
        def_args['max_h'] = 5000
        def_args['max_best_rules'] = 300#
        def_args['param_relation_embed'] = True
        def_args['param_entity_embed'] = False
        def_args['init_weight_with_prior'] = False
        def_args['prior_coef'] = 1000#0.01
        def_args['use_neg_rules'] = False
        def_args['disable_gnd'] = False
        def_args['disable_selflink'] = False
        def_args['drop_neg_gnd'] = False
        def_args['pgnd_num'] = 256
        def_args['pgnd_selflink_rate'] = 8
        def_args['pgnd_nonselflink_rate'] = 0
        def_args['pgnd_weight'] = 0.1
        def_args['max_pgnd_rules'] = None  # def_args['max_rules']
        def_args['predictor_num_epoch'] = 100#200000
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
        def_args['rely_gen'] = True #Only relying on the generator for scoring, and the rule's pgnd only relies on topology generation.
        def_args['modelling'] = valuate_modelling #0:'one',1:'weight',2:'rotatE',3:'transE'. one：1; weight：Score as the weight of the rule (i.e. the generator's scoring); 'rotatE','transE':评分使用对应算法的score.
        def_args['Auto_best'] = Auto_best #0:last model，1:self-adaption，2:best when testing.
        def_args['inductive'] = inductive

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

            self._args[k] = str(v) if make_str(v) else v

    def arg(self, name, apply=None):

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
        if self.arg('modelling') == 2:#'rotatE'
            _rule_embed = self.rule_embed()#shape = (self.num_rule, self.rotate.embed_dim)
        elif self.arg('modelling') == 3 and self.em != 0:#TODO 'transE'
            _rule_embed = torch.stack(self.rule_emb).cuda()
        rule_embed = []
        crule = []
        crule_weight = []
        centity = []
        cweight = []
        csplit = [0]

        for single in batch:

            _h, _, _, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) == 0:
                csplit.append(csplit[-1])
                continue
            crule.append(_crule + len(centity) * self.num_rule)
            crule_weight.append(rule_weight.index_select(0, _crule))#len=pgnd
            centity.append(_centity)
            cweight.append(_cweight)
            if self.arg('modelling') == 2:#'rotatE'
                rule_embed.append(self.rotate.embed(_h, _rule_embed))#hOr,(self.num_rule, self.rotate.embed_dim)
            elif self.arg('modelling') == 3 and self.em != 0:
                h_embed = self.e_emb[_h].squeeze().cuda()
                #print(h_embed.shape,_rule_embed.shape)
                t_embed = h_embed+_rule_embed
                rule_embed.append(t_embed)
                #print(h_embed.shape,t_embed.shape)
            csplit.append(csplit[-1] + _crule.size(-1))


        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            crule_weight = torch.tensor([]).cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).cuda()
            rule_embed = torch.tensor([]).cuda()
            cscore = torch.tensor([]).cuda()
            cscore_drule = torch.tensor([]).cuda()
        else:
            crule = torch.cat(crule, dim=0)
            crule_weight = torch.cat(crule_weight, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)
            if self.arg('modelling') == 2:#'rotatE'
                rule_embed = torch.cat(rule_embed, dim=0)#
                cscore_drule = self.cscore(rule_embed, crule, centity, cweight)#
                cscore = cscore_drule * crule_weight#
            elif self.arg('modelling') == 0 or self.em==0:#'one',or em0,not need embedding
                cscore_drule = torch.ones(centity.shape[0]).float().cuda()/10
                cscore = cscore_drule * crule_weight
            else:#TODO 'transE'
                rule_embed = torch.cat(rule_embed, dim=0)#
                cscore_drule = cos(self.e_emb[centity].T,rule_embed[crule].T)#len = number of all pgnd
                cscore = cscore_drule * crule_weight


        loss = torch.tensor(0.0).cuda().requires_grad_() + 0.0

        loss_rule = 0.0
        result = []
        augfix = 10000
        for i, single in enumerate(batch):
            _h, t_list, mask, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) != 0:
                crange = torch.arange(csplit[i], csplit[i + 1]).cuda()
                sparse_score = torch.sparse.FloatTensor(
                    torch.stack([_centity, _crule], dim=0),
                    self.index_select(cscore, crange),
                    torch.Size([E, self.num_rule])
                )
                score = torch.sparse.sum(sparse_score, -1).to_dense()#shape = (E,1)
                if individual_rule:
                    score_rule = torch.sparse.FloatTensor(
                    torch.stack([_centity, _crule], dim=0),
                    self.index_select(cscore_drule, crange),
                    torch.Size([E, self.num_rule])).to_dense()#shape = (E,num_rule)
            else:
                score = torch.zeros(self.E).cuda()
                score.requires_grad_()
                if individual_rule:
                    score_rule = torch.zeros(self.E,self.num_rule).cuda()


            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()
                score = self.index_select(score, ans_can)
                
                if individual_rule:


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


            if self.recording:
                self.record.append((score.cpu(), mask, t_list))

            elif not self.training:#valid
                for t in t_list:
                    result.append(self.metrics.apply(score, mask.bool(), t))#hit@1、mrr
            
            if score.dim() == 0:
                continue

            score = score.softmax(dim=-1)#shape = (E,1)
            neg = score.masked_select(~mask.bool())#shape = (E,1)

            loss += neg.sum()

            for t in t_list:
                s = score[t]
                wrong = (neg > s)
                loss += ((neg - s) * wrong).sum() / wrong.sum().clamp(min=1)
            
            if individual_rule:

                with torch.no_grad():

                    sc_rule = score_rule#shape = (E,num_rule)
                    mask_rule = mask.unsqueeze(-1).repeat(1,self.num_rule)

                    ng_rule = (sc_rule*~mask_rule)

                    loss_rule += ng_rule.sum(0)*augfix



                    for t in t_list:

                        ls_ng =augfix*((ng_rule - sc_rule[t]) * (ng_rule > sc_rule[t])).sum(0) 
                        ls_f =  (ng_rule > sc_rule[t]).sum(0).clamp(min=1)

                        loss_rule += ls_ng / ls_f
        self.t_list = t_list
        return loss / len(batch),loss_rule/len(batch), self.metrics.summary(result)

    def _evaluate(self, valid_batch, batch_size=None):
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')#4
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')


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

    def _make_batch(self, h, t_list, answer=None, rgnd_buffer=None):
        # print("make_batch in")
        istest = True
        if answer is None:
            answer = t_list
        
        if rgnd_buffer is None :
            istest = False
            rgnd_buffer = self.rgnd_buffer
        crule = []
        centity = []
        cweight = []
        gnd = []
        max_pgnd_rules = self.arg('max_pgnd_rules')
        if max_pgnd_rules is None:
            max_pgnd_rules = self.arg('max_rules')#1000
        for i, rule in enumerate(self.rules_exp):
            if i != 0 and not self.arg('disable_gnd'):
                key = (h, self.r, rule)
                if key in rgnd_buffer:
                    rgnd = rgnd_buffer[key]
                else:

                    if not istest :#train\valid
                        rgnd = calc_groundings(h, rule,self.r_groundings)
                    else:#test
                        rgnd = groundings.groundings(h, rule)

                    ans_can = self.arg('answer_candidates', apply=lambda x: x)#
                    if ans_can is not None:
                        ans_can = set(ans_can.cpu().numpy().tolist())
                        rgnd = list(filter(lambda x: x in ans_can, rgnd))#
                    rgnd_buffer[key] = rgnd

                ones = torch.ones(len(rgnd))
                centity.append(torch.LongTensor(rgnd))
                crule.append(ones.long() * i)
                cweight.append(ones)
            else:
                rgnd = []

            gnd.append(rgnd)
            if i == 0 and self.arg('disable_selflink'):
                continue
            if i >= max_pgnd_rules:
                continue
            num = self.arg('pgnd_num') * self.arg('pgnd_selflink_rate' if i == 0 else 'pgnd_nonselflink_rate')

            
            if self.arg('rely_gen'):pgnd = torch.LongTensor(gnd[i])#依赖生成器规则，仅考虑规则直接产生的groundings
            else:pgnd = self.pgnd(h, i, num, gnd[i])
            

            ones = torch.ones(len(pgnd))
            centity.append(pgnd.long().cpu())
            crule.append(ones.long() * i)
            cweight.append(ones * self.arg('pgnd_weight'))

        # print("iter done")
        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).float().cuda()
        else:
            crule = torch.cat(crule, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)

        #################
        # print("work", answer)

        # print("make_batch out")

        return h, t_list, list2mask(answer, self.E), crule, centity, cweight

    # make all batchs
    def make_batchs(self, init=False):
        print = self.print
        # if r is not None:
        # 	self.r = r
            
        dataset = self.dataset

        graph = build_graph(dataset['train'], self.E, self.R)
        def filter(tri):
            a = defaultdict(lambda: [])
            for h, r, t in tri:
                if r == self.r:
                    a[h].append(t)
            return a

        train = filter(dataset['train'])
        valid = filter(dataset['valid'])
        test = filter(dataset['test'])

        #For inductive test
        if inductive:
            dataset = self.org_dataset

        ind_test = []
        for _h,_r,_t in dataset["test"]:
            if _r == self.r:
                continue
            ind_test.append([_h,_r,_t])

        if exp=='ilpc' or exp == 'ilpc-large':
              ind_test = ind_test + dataset["inference"]

        # if self.arg('inductive'):
        #      graph_test = build_graph(dataset['train'] + dataset['valid'] + ind_test, self.E, self.R)
        # else:
        #     graph_test = build_graph(dataset['train'] + dataset['valid'], self.E, self.R)
        #graph_test = build_graph(dataset['train'] + dataset['valid'], self.E, self.R)
        graph_test = build_graph(dataset['train'] + dataset['valid'] + ind_test, self.E, self.R)
        #graph_test = build_graph(dataset['train'] + dataset['valid'] + dataset["test"], self.E, self.R)
        

        answer_valid = defaultdict(lambda: [])
        answer_test = defaultdict(lambda: [])
        for a in [train, valid]:
            for k, v in a.items():
                answer_valid[k] += v
                answer_test[k] += v
        for k, v in test.items():
            answer_test[k] += v#

        if len(train) > self.arg('max_h'):
            from random import shuffle
            train = list(train.items())
            shuffle(train)
            train = train[:self.arg('max_h')]
            train = {k: v for (k, v) in train}

        print_epoch = self.arg('predictor_init_print_epoch')

        self.train_batch = []
        self.valid_batch = []
        self.test_batch = []


        if init:
            
            groundings.use_graph(graph)
            def gen_init(self, train, print_epoch):
                for i, (h, t_list) in enumerate(train.items()):
                    if i % print_epoch == 0:
                        print(f"init_batch: {i}/{len(train)}")
                    yield self._make_batch(h, t_list)

            return gen_init(self, train, print_epoch)
        else:
            groundings.use_graph(graph_test)
            for i, (h, t_list) in enumerate(test.items()):
                if i % print_epoch == 0:
                    print(f"test_batch: {i}/{len(test)}")
                self.test_batch.append(
                    self._make_batch(h, t_list, answer=answer_test[h], rgnd_buffer=self.rgnd_buffer_test))
                
            groundings.use_graph(graph)
            #_make_batch：h, t_list, list2mask(answer, self.E), crule, centity, cweight
            for i, (h, t_list) in enumerate(train.items()):
                if i % print_epoch == 0:
                    print(f"train_batch: {i}/{len(train)}")
                batch = list(self._make_batch(h, t_list))
                for t in t_list:
                    batch[1] = [t]
                    self.train_batch.append(tuple(batch))

            for i, (h, t_list) in enumerate(valid.items()):
                if i % print_epoch == 0:
                    print(f"valid_batch: {i}/{len(valid)}")
                self.valid_batch.append(self._make_batch(h, t_list, answer=answer_valid[h]))

            
            

    # Make batchs for generator, used in M-step
	# weight：Hscore;
    def make_gen_batch(self, generator_version=1):
        self.tmp__rule_embed = self.rule_embed()
        weight = torch.zeros_like(self.rule_weight_raw).long().cuda()
        for i, batch in enumerate(self.train_batch):
            cho = self.best_rules(batch)
            weight[cho] += len(batch[1]) 

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

        def valid():
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
        if self.em == 0:
            num_epoch = 0
        print(f"Candidate rules:{self.rules_exp[:10]}/{len(self.rules_exp)}rules.")
        for epoch in range(1, num_epoch + 1):
            if epoch % max(1, len(train_batch) // batch_size) == 0:
                from random import shuffle
                shuffle(train_batch)
            batch = [train_batch[(epoch * batch_size + i) % len(train_batch)] for i in range(batch_size)]

            if epoch % print_epoch == 0:self.train_print = True
            loss,loss_rule = train_step(batch)
            cum_loss += loss.item()
            ret_loss += loss.item()
            ret_loss_rule = ret_loss_rule+loss_rule
            self.train_print = False
            if epoch % print_epoch == 0:#
                print(epoch,len(batch),"tail:",self.t_list)
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_predictor #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")

                if isinstance(ret_loss_rule,list):
                    print(f"ret_loss_rule = {ret_loss_rule[:10]}")
                else:
                    print(f"ret_loss_rule = {ret_loss_rule}")
                cum_loss *= 0

            if epoch % valid_epoch == 0:
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

        with torch.no_grad():
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

        print("__V__\t" + ("\t".join([str(self.r), str(int(best[0]))] + list(map(lambda x: "%.4lf" % x, best[1:])))))
        print("__T__\t" + ("\t".join([str(self.r), str(int(test[0]))] + list(map(lambda x: "%.4lf" % x, test[1:])))))
        #test[0]: sum of t_list, test[1:]: mr,mrr,h1,h3,h10
        if self.em>0:
            if self.arg('Auto_best')==2:
                if self.bestMRR < float(test[2]):
                    self.result_sum.append(test)
                    self.bestMRR = float(test[2])
            elif  self.arg('Auto_best')==1:
                if self.bestMRR <= float(best[2]):
                    self.result_sum.append(test)
                    self.bestMRR = float(best[2])
            else:
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
        self.path_gnd = []#[[{h:count,...},{gnd:count,...},...{t:count,...}],...]
        #self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.prec = []
        self.rule_emb = []
        self.r_groundings = None # buffer Only for train graph
        
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

    def loss(self, inputs, target, mask, weight):
        return 0
        
    

    def train_model(self,r, ret_loss,ret_loss_rule,em_epoch = 1,num_epoch=1, num_em_epoch=100):
        print = self.print

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

        def rule_loss(ret_loss_rule,rule_path,path_gnd,metric_final = True):#XXX train emb,train graph,current gnd
            num_rule = len(rule_path)
            if metric_final:
                metric_e_emb = self.e_emb #dim=200
                metric_emb_org = self.r_emb
            else:
                metric_e_emb = self.model_gat.entity_embeddings #dim=200
                metric_emb_org = self.model_gat.relation_embeddings

            inv_metric_r_emb = - metric_emb_org

            metric_emb = torch.cat([metric_emb_org,inv_metric_r_emb])

            path_loss = []
            target_r = self.r
            #Cut wrong path;TODO C++ implementation,add(h,t) to rule sample
            #t_list = self.Rt[target_r]
            path_emb_list = []
            
            if isinstance(ret_loss_rule,list):
                print(f"ret_loss_rule = {ret_loss_rule[:10]}")
            else:
                print(f"ret_loss_rule = {ret_loss_rule}")
            for i,path in enumerate(rule_path):
                if i<1:
                    continue
                rule = list(path)
                eval_loss = ret_loss_rule[i].item()/rule_loss_aug+1e-3
                gnd_list = path_gnd[i][1:-1]#TODO,add cb_r:h_list,t_list
                if i < 10:
                    print(f"Rule {rule} evaluator loss {eval_loss}.Length=",len(gnd_list),len(rule))
                if path_model != 5:
                    assert len(gnd_list)==len(rule)-1
                
                path_emb = 0
                if path_model ==2:
                    if path_gnd[i][0] is not None:
                        h_emd = metric_e_emb[list(path_gnd[i][0].keys()),:].cuda()
                        h_emd_mean = h_emd.mean(dim=0)
                        path_emb = 0-h_emd_mean
                    else:
                        print("Error.No head.")
                last_r = -1
                ln = len(rule)
                for j,r in enumerate(rule):
                    if last_r>=0:
                        if path_model == 5: continue
                        if last_r not in self.r_dist.keys():
                            print("Error last_r",last_r)
                        elif self.r_dist[last_r] is None:
                            print("Error last_r",last_r)
                        else:
                            if self.r_dist[last_r][r] is not None:
                                if path_model ==3:
                                    path_emb = infer(path_emb,(ln-j)*self.r_dist_path(last_r,r,gnd_list[j-1],metric_final),model=model_type)
                                else:
                                    path_emb = infer(path_emb,self.r_dist_path(last_r,r,gnd_list[j-1],metric_final),model=model_type)
                            else:
                                print("Error last_r&r",last_r,r)
                    if r < self.R:
                        if path_model ==3 or path_model == 4:
                            if j==0:path_emb = (ln)*metric_emb[r]
                            else:
                                path_emb = chain(path_emb,(ln)*metric_emb[r],model=model_type)
                        else:
                            if j==0:path_emb = metric_emb[r]
                            else:
                                path_emb = chain(path_emb,metric_emb[r],model=model_type)
                        if path_model ==2:
                            if j==0:
                                path_emb = chain(path_emb,metric_emb[r],model=model_type)
                            if j==ln-1:
                                path_emb = chain(path_emb,-1*metric_emb[r],model=model_type)
                    else:
                        if path_model ==3 or path_model == 4:
                            if j==0:path_emb = (-ln)*metric_emb[r]
                            else:
                                path_emb = chain(path_emb,(-ln)*metric_emb[r],model=model_type)
                        else:
                            if j==0:path_emb = -1*metric_emb[r]
                            else:
                                path_emb = chain(path_emb,-1*metric_emb[r],model=model_type)
                        if path_model ==2:
                            if j==0:
                                path_emb = chain(path_emb,-1*metric_emb[r],model=model_type)
                            if j==ln-1:
                                path_emb = chain(path_emb,metric_emb[r],model=model_type)
                    last_r = r
                if path_model ==2:
                    if path_gnd[i][-1] is not None:
                        wt_t = torch.tensor(list(path_gnd[i][-1].values()),dtype=torch.float32)
                        wt_t_norm = torch.nn.functional.normalize(wt_t,p=1,dim=0).cuda() 
                        t_emd = metric_e_emb[list(path_gnd[i][-1].keys()),:].cuda()
                        t_wt = t_emd*wt_t_norm.unsqueeze(-1)
                        t_emd_mean = t_wt.sum(dim=0)

                        path_emb = infer(path_emb,t_emd_mean,model=model_type)
                    else:     
                        print("Error.No tail.")
                if path_model == 1 or path_model == 4 or path_model ==5:
                    pathnum = 1
                else:
                    pathnum = 2
                #path_emb_list.append(path_emb)
                #sim = (1-cos(path_emb/2,metric_emb[target_r]))/eval_loss
                #scale = ln*(ln-1)/2
                #sim = (torch.linalg.norm(path_emb-metric_emb[target_r],ord=1))/eval_loss
                #sim_model = 'cos'
                # if sim_model == 'cos':
                #     sim = cos(path_emb/pathnum,metric_emb[target_r])/eval_loss
                #     #sim = (1-cos(path_emb/2,metric_emb[target_r]))/eval_loss                
                # elif sim_model == 'F2':
                #     sim = torch.linalg.norm((path_emb/pathnum)-metric_emb[target_r],ord=2)/eval_loss #ord=2
                # else:#F1
                #     sim = torch.linalg.norm((path_emb/pathnum)-metric_emb[target_r],ord=1)/eval_loss #ord=2
                sim = dist(path_emb/pathnum,metric_emb[target_r],model = model_type,sim_model = sim_model)/eval_loss
                path_loss.append((sim).unsqueeze(-1)) 
            # path_emb_all = torch.stack(path_emb_list)#(num_rule,emb_size)
            # target_emb = metric_emb[target_r].unsqueeze(0).repeat(self.num_rule,1)
            # y = torch.ones(len(num_rule)).cuda()
            # loss = gat_loss_func(path_emb_all, target_emb, y)
            #loss_norm = torch.norm(loss, p=1, dim=1)
            print("path_loss[:10]",path_loss[:10])
            loss = torch.cat(path_loss)
            return loss.mean()


        
        cum_loss = 0
        epoch_losses = []
        init_flag = True
        em_weight = 0.1
        for epoch in range(1, num_epoch + 1):
            print("\nepoch-> ", epoch)
            random.shuffle(Corpus_.train_triples)
            random.shuffle(cb_Corpus_.train_triples)
            Corpus_.train_indices = np.array(
                list(Corpus_.train_triples)).astype(np.int32) 
            cb_Corpus_.train_indices = np.array(
                list(cb_Corpus_.train_triples)).astype(np.int32)

            model_gat.train()  # getting in training mode
            cb_model_gat.train()
            start_time = time.time()
            epoch_loss = []
            if len(Corpus_.train_indices) < len(cb_Corpus_.train_indices):
                if len(Corpus_.train_indices) % args.batch_size_gat == 0:
                    num_iters_per_epoch = len(
                        Corpus_.train_indices) // args.batch_size_gat
                else:
                    num_iters_per_epoch = (
                        len(Corpus_.train_indices) // args.batch_size_gat) + 1
            else:
                if len(cb_Corpus_.train_indices) % args.batch_size_gat == 0:
                    num_iters_per_epoch = len(
                        cb_Corpus_.train_indices) // args.batch_size_gat
                else:
                    num_iters_per_epoch = (
                        len(cb_Corpus_.train_indices) // args.batch_size_gat) + 1
            print("\n num_iters_per_epoch ==", num_iters_per_epoch)
            print("\n len(Corpus_.train_indices) ==", len(Corpus_.train_indices))
            print("\n len(cb_Corpus_.train_indices) ==", len(cb_Corpus_.train_indices))

            for iters in range(num_iters_per_epoch):
                print("\n iters-> ", iters)
                start_time_iter = time.time()
                train_indices, train_values = Corpus_.get_iteration_batch(iters)
                cb_train_indices, cb_train_values = cb_Corpus_.get_iteration_batch(iters)
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
                #self.e_emb = self.model_gat.final_entity_embeddings
                self.r_emb = self.model_gat.final_relation_embeddings
                #self.cb_e_emb  = self.cb_model_gat.final_entity_embeddings
                self.cb_r_emb = self.cb_model_gat.final_relation_embeddings
           
                opt.zero_grad()
                trs_loss = batch_gat_loss(
                    gat_loss_func, train_indices, entity_embed, relation_embed)
                cb_loss = batch_gat_loss(
                    gat_loss_func, cb_train_indices, cb_entity_embed, cb_relation_embed,mod = 1)
                rl_loss = rule_loss(ret_loss_rule,self.rules_path,self.path_gnd,metric_final = True)#torch.nn.CosineEmbeddingLoss(reduction = "mean")
                #loss = (1-em_weight)*(loss + cb_loss)+10*em_weight*rl_loss #+ em_weight*ret_loss
                #loss = 0.4*(trs_loss + cb_loss)+0.6*rl_loss #best
                loss = (trs_loss + cb_loss)+em_weight*rl_loss
                #loss = rl_loss
                loss.backward()
                opt.step()
                end_time_iter = time.time()
                epoch_loss.append(loss.data.item())

                print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
                
            
            sch.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            epoch_mean_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_losses.append(epoch_mean_loss)


            print(f"train_generator #{epoch} epoch_mean_loss = {epoch_mean_loss}")
        # if em_epoch%10==0 or em_epoch == num_em_epoch - 1:
        #     save_model(model_gat, args.data, em_epoch,
        #             args.output_folder,prex = 'em_e')
        #     save_model(cb_model_gat, args.data, em_epoch,
        #            args.output_folder,prex = 'em_r')
            
        if torch.cuda.device_count() > 1:
            self.model_gat = model_gat.module
            self.cb_model_gat = cb_model_gat.module
        else:
            self.model_gat = model_gat
            self.cb_model_gat = cb_model_gat

    # gen rules from rule_file
    def rule_gen(self,r,rule_file,num_samples, max_len,modelling = 3):#XXX train emb,train graph,current gnd

        rules = []
        rule_set = set([tuple(), (r,)])

        #update r_dist
        self.e_emb = self.model_gat.final_entity_embeddings
        self.r_emb = self.model_gat.final_relation_embeddings
        #self.cb_e_emb  = self.cb_model_gat.final_entity_embeddings
        self.cb_r_emb = self.cb_model_gat.final_relation_embeddings
        #self.r_dist = self.patch_r_dist(r_num = 2*self.R)
        has_inv = False
        with open(rule_file) as file:
            for i, line in enumerate(file):
                #try:
                path, prec = line.split('\t')
                path = tuple(map(int, path.split()))
                prec_ = float(prec.split()[0])
                if prec_> 1 or prec_<0 :
                    # to avoid negative and nan
                    prec_ = 0.0001
                recall_ = float(prec.split()[1])
                if recall:
                    prec = prec_ * recall_* 100
                else:
                    prec = prec_
                if not (prec >= 0.0001):
                    # to avoid negative and nan
                    prec = 0.0001
                if path in rule_set:
                    continue
                if not inv_relatation:
                    for rel in path:
                        if rel >= self.R and rel != r:#237
                            has_inv = True
                            break
                    if has_inv:
                        has_inv = False
                        #print("Jump inv_rel rule:",path)
                        continue

                gnd_list = []
                sc ,gnd_list = self.cb_rule_score_gnd(r,path)
                print(path,"Score  = ",sc,"sample prec",prec)
                rule_set.add(path)
                if len(path) <= max_len:
                    rules.append((path,gnd_list, sc,i,prec))
                    #self.path_gnd.append((gnd_list, sc,i))
                # except:
                #     continue          
        rules = sorted(rules, key=lambda x: (x[2], x[3]), reverse=True)[:num_samples]#
        rules = [((r,),[dict()], 1, -1,1)]+rules
                

        print(f"CbGAT generate candidate rules: |rules| = {len(rules)} max_rule_len = {max_len}")
        x = torch.tensor([sc for _,_, sc,_,_  in rules]).cuda()

        prior = x
        self.path_gnd = [gnd_list for _,gnd_list, _, _,_ in rules]#[[{h:count,...},{gnd:count,...},...{t:count,...}],...]
        self.prec = [prec for _,_, _, _,prec in rules]
        rules = [path for path, _,_, _,_ in rules]

        if modelling==3:#transE
            for path in rules:            
                path_emb = self.cb_rule(self.r,path)
                self.rule_emb.append(path_emb)

        return rules,prior
    
    def cb_rule(self,target_r,path,obj = 'emb'):
        
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
        if obj=='ang':
            output = cos(path_emb,self.r_emb[target_r])#(1,-1)
            #angs = (torch.acos(output)*180/3.1415926).item()
            angs = output.item()
            #print("Score for rule",rule_path," = ",angs)
            return angs
        else:
            return path_emb
    
    def cb_rule_score_gnd(self,target_r,path,metric_final = True):
        gnd_list = []
        if metric_final:
            metric_e_emb = self.e_emb #dim=200
            metric_r_emb_org = self.r_emb
        else:
            metric_e_emb = self.model_gat.entity_embeddings #dim=100
            metric_r_emb_org = self.model_gat.relation_embeddings #dim=100
        inv_metric_r_emb = - metric_r_emb_org

        metric_r_emb = torch.cat([metric_r_emb_org,inv_metric_r_emb])

        h_list = list(self.Rh[target_r])#XXX  train graph
        hgnd = dict()
        for h in h_list:
            hgnd[h]=1
        if len(hgnd)>0:
            gnd_list.append(hgnd)
        else:
            gnd_list.append(None)
        path_emb = torch.zeros(metric_r_emb.shape[-1]).cuda()
        if path_model ==2:
            if gnd_list[0] is not None:
                h_emd = metric_e_emb[list(hgnd.keys()),:].cuda()
                h_emd_mean = h_emd.mean(dim=0)
                path_emb = -h_emd_mean  #dim=200
            else:
                #path_emb = torch.zeros(metric_r_emb.shape[-1]).cuda()
                print("Error.No head.")

        rule_path = list(path)
        last_r = -1
        cur_path = []
        gnd = None
        ln = len(rule_path)
        for j,r in enumerate(rule_path):
            if last_r>=0:                
                if path_model == 5: continue
                if last_r not in self.r_dist.keys():
                    gnd_list.append(None)
                    print("Error key.last_r:",last_r)
                    return 1e-8,gnd_list
                elif self.r_dist[last_r] is None:
                    gnd_list.append(None)
                    print("Error None value.last_r:",last_r)
                    return 1e-8,gnd_list
                else:
                    if self.r_dist[last_r][r] is not None:
                        gnd = h_path_t(h_list,cur_path,self.r_groundings) #current graph;self.r_groundings
                        gnd_list.append(gnd)

                        if path_model ==3:
                            path_emb = infer(path_emb,(ln-j)*self.r_dist_path(last_r,r,gnd,metric_final),model=model_type)
                        else:
                            path_emb = infer(path_emb,self.r_dist_path(last_r,r,gnd,metric_final),model=model_type)#XXX train emb,train graph,current gnd
                    else:
                        gnd_list.append(None)
                        print("Error None value.last_r:",last_r,"r:",r)
                        return 1e-8,gnd_list
            if r < self.R:              
                if path_model ==3 or path_model == 4:
                    if j==0:path_emb = (ln)*metric_r_emb[r]
                    else:
                        path_emb = chain(path_emb,(ln)*metric_r_emb[r],model=model_type)
                else:
                    if j==0:path_emb = metric_r_emb[r]
                    else:
                        path_emb = chain(path_emb,metric_r_emb[r],model=model_type)
                if path_model == 2:
                    if j==0:
                        path_emb = chain(path_emb,metric_r_emb[r],model=model_type)
                    if j==ln-1:
                        path_emb = chain(path_emb,-1*metric_r_emb[r],model=model_type)
            else:               
                if path_model ==3 or path_model == 4:
                    if j==0:path_emb = (-ln)*metric_r_emb[r]
                    else:
                        path_emb = chain(path_emb,(-ln)*metric_r_emb[r],model=model_type)
                else:
                    
                    if j==0:path_emb = -1*metric_r_emb[r]
                    else:
                        path_emb = chain(path_emb,-1*metric_r_emb[r],model=model_type)
                if path_model == 2:
                    if j==0:
                        path_emb = chain(path_emb,-1*metric_r_emb[r],model=model_type)
                    if j==ln-1:
                        path_emb = chain(path_emb,metric_r_emb[r],model=model_type)
                        
            last_r = r
            cur_path.append(r)
        tgnd = h_path_t(h_list,cur_path,self.r_groundings)#current graph
        if len(tgnd)>0:
            gnd_list.append(tgnd)
        else:
            gnd_list.append(None)
        if path_model == 2:
            if gnd_list[-1] is not None:
                wt_t = torch.tensor(list(tgnd.values()),dtype=torch.float32)
                wt_t_norm = torch.nn.functional.normalize(wt_t,p=1,dim=0).cuda() 

                t_emd = metric_e_emb[list(tgnd.keys()),:].cuda()
                t_wt = t_emd*wt_t_norm.unsqueeze(-1)
                t_emd_mean = t_wt.sum(dim=0)

                path_emb = infer(path_emb,t_emd_mean,model=model_type)
            else:     
                print("Error.No tail.")
        
        if path_model == 1 or path_model == 4 or path_model == 5:
            pathnum = 1
        else:
            pathnum = 2

        #sim_model = 'F1'
        if model_type=='rotatE':            
            output = 100/dist(path_emb/pathnum,metric_r_emb[target_r],model = model_type,sim_model = sim_model)#(1,-1)  
        elif sim_model == 'cos' :
            output = dist(path_emb/pathnum,metric_r_emb[target_r],model = model_type,sim_model = sim_model)#(1,-1)            
        elif sim_model == 'F2':
            #output = 100/torch.linalg.norm((path_emb/pathnum)-metric_r_emb[target_r],ord=2) #ord=2
            output = 100/dist(path_emb/pathnum,metric_r_emb[target_r],model = model_type,sim_model = sim_model)
        else:#F1
            output = 100/dist(path_emb/pathnum,metric_r_emb[target_r],model = model_type,sim_model = sim_model)
        angs = output.item()

        # print("Score for rule",rule_path," = ",angs)
        return angs,gnd_list

    
    def r_dist_path(self,last_r,r,gnd,metric_final = True):

        if metric_final:
            metric_emb = self.cb_r_emb #dim=200
        else:
            metric_emb = self.cb_model_gat.relation_embeddings #dim=50
        
        mide =  self.cb_kg[last_r][r]

        link = dict()
        for k,v in gnd.items():
            try:
                if k in mide:
                    if v > 100:
                        v = 100
                    link[k]=v
            except:
                print("ERROR k in mide",len(mide))
                print(k,v)
                print(mide)
                continue
        #print("Finish gndlink.k,v,len(link)",k,v,len(link))
        # if last_r ==0:
        #     print("len(self.cb_kg[last_r][r]),len(gnd),len(cb_r_list):",len(mide),len(gnd),len(link))
        #     print(f"cb_kg[{last_r}][{r}]:",mide)
        wt = torch.tensor(list(link.values()),dtype=torch.float32)
        wt_norm = torch.nn.functional.normalize(wt,p=1,dim=0).cuda() 
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

            if r < self.R:
                path_emb = path_emb + self.r_emb[r]
            else:
                path_emb = path_emb - self.r_emb[r-self.R]
            last_r = r
        output = cos(path_emb,self.r_emb[target_r])
        angs = output.item()
        # print("Score for rule",rule_path," = ",angs)
        return angs

    def init(self):
        model_gat,cb_model_gat = read_cd_gat(args)
        self.model_gat = model_gat
        self.cb_model_gat = cb_model_gat
        
        self.e_emb = self.model_gat.final_entity_embeddings
        self.r_emb = self.model_gat.final_relation_embeddings#
        self.cb_r_emb = self.cb_model_gat.final_relation_embeddings#
        self.r_emb_len = self.r_emb.shape[-1]

        self.R = self.r_emb.shape[0]

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
                    mid_e = torch.Tensor(self.cb_kg[rh][next_r]).long()
                    mid_e_emd = self.cb_r_emb[mid_e,:]
                    mean_mid_e_emd = mid_e_emd.mean(dim = 0)
                    dist_r[rh][next_r] = mean_mid_e_emd
                else:
                    dist_r[rh][next_r] = None
        return dist_r

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

    epoch_load = 0
    epoch_load = args.epochs_gat

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT) #GAT

    pre  = 'cb_e'
    if epoch_load>0:
        model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_e{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)

    cb_model_gat = SpKBGATModified(model_gat.relation_embeddings, cb_relation_embeddings, args.entity_out_dim, args.entity_out_dim,
    args.drop_GAT, args.alpha, args.nheads_GAT,cb_flag=True) #cubic GAT

    pre  = 'cb_r'
    if epoch_load>0:
        cb_model_gat.load_state_dict(cleanup_state_dict(torch.load(
        '{}/trained_cb_r{}.pth'.format(args.output_folder, epoch_load - 1))), strict=True)

    
    if CUDA:
        model_gat.cuda()
        cb_model_gat.cuda()
        # if torch.cuda.device_count() > 1:
        #     print("Use", torch.cuda.device_count(), 'gpus')
        #     model_gat = nn.DataParallel(model_gat, device_ids=[torch.cuda.current_device()])
        #     cb_model_gat = nn.DataParallel(cb_model_gat, device_ids=[torch.cuda.current_device()])

    return model_gat,cb_model_gat
