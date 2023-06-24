from collections import defaultdict

import rotate_compare_cppext
import rotate_dist_cppext
import torch
from main import reverse
from reasoning_model import ReasoningModel


class RotatE(ReasoningModel):#内含一个静态的c++类库，存放了rotateE的计算数据

    def __init__(self, dataset, pretrained=None):
        # load pretrained rotate
        super(RotatE, self).__init__()

        E = dataset['E']
        R = dataset['R']
        self.E = E
        self.R = R

        self.answer = defaultdict(lambda: [])
        self.answer_test = defaultdict(lambda: [])

        for item in ['train', 'test', 'valid']:
            for (h, r, t) in dataset[item]:
                if item != 'test':
                    self.answer[(h, r)].append(t)

                self.answer_test[(h, r)].append(t)

        if pretrained is None:
            self.gamma = 0.0
            self.embed_dim = 1
            self.embed_range = 1.0
            self.entity_embed = torch.zeros(E, 2).float()
            self.relation_embed = torch.zeros(R, 1).float()
        else:
            import numpy
            import json
            config = json.load(open(f"{pretrained}/config.json"))
            self.gamma = config['gamma']
            self.embed_dim = config['hidden_dim']
            self.embed_range = (self.gamma + 2.0) / self.embed_dim
            self.entity_embed = torch.tensor(numpy.load(f"{pretrained}/entity_embedding.npy"))
            relation_embed = torch.tensor(numpy.load(f"{pretrained}/relation_embedding.npy"))
            if reverse:self.relation_embed = (torch.cat([-relation_embed, relation_embed], dim=0))
            else:
                self.relation_embed = (torch.cat([relation_embed, -relation_embed], dim=0))

        # pi = 3.141592653589793238462643383279
        # self.relation_embed = self.relation_embed / self.embed_range * pi

        self.entity_embed = self.entity_embed.cuda()
        self.relation_embed = self.relation_embed.cuda()
        # self._tmp = torch.nn.Parameter(torch.zeros(1))
        self.cuda()

    def _attatch_empty_relation(self):
        return torch.cat([self.relation_embed, torch.zeros(self.embed_dim).cuda().unsqueeze(0)], dim=0)

    @staticmethod
    def dist(*args):
        return RotatEDist.apply(*args)

    @staticmethod
    def compare(*args):
        return RotatECompare.apply(*args)

    def embed(self, h_embed, r_embed):
        if isinstance(h_embed, int):
            h_embed = self.entity_embed.index_select(0, torch.tensor(h_embed).cuda()).squeeze().cuda()
        if isinstance(r_embed, int):
            r_embed = self.relation_embed.index_select(0, torch.tensor(r_embed).cuda()).squeeze().cuda()

        re_h, im_h = torch.chunk(h_embed, 2, dim=-1)#根据rotateE的原理，最后一维前后两段分别存放向量的实部和虚部

        pi = 3.141592653589793238462643383279
        r_embed = r_embed / (self.embed_range / pi)##rotateE的关系向量存放的是旋转角度信息
        re_r = torch.cos(r_embed)
        im_r = torch.sin(r_embed)

        re_res = re_h * re_r - im_h * im_r#实部
        im_res = re_h * im_r + im_h * re_r#虚部

        return torch.cat([re_res, im_res], dim=-1)

    def infer(self, infer_tris, valid=False, graph=None):
        self.entity_embed = self.entity_embed.cuda()
        self.relation_embed = self.relation_embed.cuda()

        results = []
        metrics = self.metrics
        with torch.no_grad():
            for i, (h, r, t) in enumerate(infer_tris):
                score = self.gamma - self.dist(self.embed(h, r), self.entity_embed)#rotateE的算法loss为（hOr-t），因此这里对所有实体打分，衡量其与tail的匹配度
                answer = (self.answer if valid else self.answer_test)[(h, r)]
                results.append(metrics.apply(score, answer, t))
            # print(h, r, t, answer)
            # print(score)

        return results

##对每个三元组样本均调用一次。
class RotatECompare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, pa, pb):#rule_embed（来自生成器）, self.rotate.entity_embed, crule, centity，entity_embed可以用指针调用
        a = a.contiguous()
        b = b.contiguous()
        pa = pa.contiguous()
        pb = pb.contiguous()
        # print("RotatE compare", a.size(), b.size())
        # print(pa.min().item(), pa.max().item())
        # print(pb.min().item(), pb.max().item())

        ctx.save_for_backward(a, b, pa, pb)
        # 针对当前三元组，batch中包含当前评估器下各个规则判断的每个tail实体（即pgnd的数量，等于 crule, centity的长度），这些对tail的判断存于centity，对应的规则存于crule
        # 本函数比较各个crule的rotateE推理结果与centity的相似度，计算其hOr，与所有实体向量的欧式距离相似度（用于判断哪些是tail），返回一个长度为pgnd的数量的数组
        return rotate_compare_cppext.forward(a, b, pa, pb)
    @staticmethod
    def backward(ctx, ogd):
        a, b, pa, pb = ctx.saved_tensors
        return rotate_compare_cppext.backward(a, b, pa, pb, ogd)


class RotatECompare_Force:
    @staticmethod
    def apply(a, b, pa, pb):
        # print("Warning: Force version used")
        a = a.contiguous()
        b = b.contiguous()
        pa = pa.contiguous()
        pb = pb.contiguous()
        a = a.index_select(0, pa)
        b = b.index_select(0, pb)
        dist = a - b
        re, im = torch.chunk(dist, 2, dim=-1)
        dist = torch.stack([re, im], dim=-1)
        dist = dist.norm(dim=-1).sum(dim=-1)
        # if pa.size(0) == 1030:
        # 	exit()
        return dist


class RotatEDist(torch.autograd.Function):#roch的函数类

    @staticmethod
    def forward(ctx, x, a):#rotateE的算法loss为（hOr-t）。这里x为输入的hOr，a为所有的实体嵌入，计算实体跟t的相似度距离
        x = x.contiguous()
        a = a.contiguous()
        dist = rotate_dist_cppext.forward(x, a)#距离度量为欧氏距离value += sqrtf(re * re + im * im)
        ctx.save_for_backward(x, a)#保存输入，用于backward
        return dist

    @staticmethod
    def backward(ctx, outgrad_dist):
        # print(outgrad_dist)
        x, a = ctx.saved_tensors
        ingrad_x, ingrad_a = rotate_dist_cppext.backward(x, a, outgrad_dist)#梯度计算，手动求导
        return ingrad_x, ingrad_a


class RotatEDist_Force:
    @staticmethod
    def apply(x, a):
        print("Warning: Force version used")
        a = x - a
        re, im = torch.chunk(a, 2, dim=-1)
        a = torch.stack([re, im], dim=-1)
        dist = a.norm(dim=-1).sum(dim=-1)
        # print(dist.size())
        return dist


class RotatEDist_Force2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        x = x.detach()
        a = a.detach()
        tmp = x - a
        re, im = torch.chunk(tmp, 2, dim=-1)
        tmp = torch.stack([re, im], dim=-1)
        dist = tmp.norm(dim=-1).sum(dim=-1)
        ctx.save_for_backward(x, a)
        # print(dist.size())
        return dist

    @staticmethod
    def backward(ctx, o):
        x, a = ctx.saved_tensors
        # print("enter backward", a.size(), x.size())

        are, aim = torch.chunk(a, 2, dim=-1)
        xre, xim = torch.chunk(x, 2, dim=-1)

        gxre = torch.zeros_like(xre).cuda()
        gxim = torch.zeros_like(xim).cuda()
        gare = torch.zeros_like(are).cuda()
        gaim = torch.zeros_like(aim).cuda()

        n = are.size(0)
        d = are.size(1)

        # print(n, d)

        for i in range(n):
            for j in range(d):
                re = xre[j] - are[i][j]
                im = xim[j] - aim[i][j]
                dis = (re ** 2 + im ** 2) ** 0.5
                # print("%d %d %.4lf %.4lf" % (i,j,dis,o[i]))
                gxre[j] += re * o[i] / dis
                gxim[j] += im * o[i] / dis
                gare[i][j] = -re * o[i] / dis
                gaim[i][j] = -im * o[i] / dis

        gx = torch.cat([gxre, gxim], dim=-1)
        ga = torch.cat([gare, gaim], dim=-1)

        # print(gx.size(), ga.size())
        return gx, ga
