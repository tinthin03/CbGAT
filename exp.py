import os,pickle

# filename="rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx300-tr9-noinv-R4+.out"
# #filename="cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx200-tr9-noinv-R4+.out"
# filename="rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx330-tr9-noinv-R4+.out"
def analysis(filename,compTnum = None,R = 237):
    with open(filename) as f:
        lines = f.readlines()
    rule_quality = dict()
    Vmrr = dict()
    Tmrr = dict()
    VH1 = dict()
    TH1 = dict()
    Tnum_dict = dict()
    roundnum = 10
    Vsmp = 0 #样本数
    Tsmp = 0 #样本数
    for i in range(roundnum):
        rule_quality[i] = 0
        Vmrr[i] = 0
        Tmrr[i] = 0
        VH1[i] = 0
        TH1[i] = 0

    #各回合的rule_quality
    rind = 0
    count_ind = 0
    tmp_qual = 0
    for line in lines:
        inf = line.strip().split(',')
        if len(inf)==4:
            #print(inf)
            if inf[1][:6]=='Top 20':
                res = inf[1].split(' = ')
                #tmp_qual = float(res[1])
                rule_quality[count_ind] += float(res[1])
        if "__V__" in line:
            inftest = line.strip().split('__V__')
            inftest = inftest[1].split('\t')
            Vnum = int(inftest[2])
            Vmrr[count_ind] += float(inftest[4])*Vnum
            VH1[count_ind] += float(inftest[5])*Vnum
        
        if "__T__" in line:
            inftest = line.strip().split('__T__')
            inftest = inftest[1].split('\t')
            Tnum = int(inftest[2])
            Tnum_dict[rind] = Tnum
            if compTnum is not None:
                if rind in compTnum.keys():
                    Tnum = compTnum[rind]
                else:
                    Tnum = 0
            Tmrr[count_ind] += float(inftest[4])*Tnum
            TH1[count_ind] += float(inftest[5])*Tnum
            count_ind += 1

        
        if count_ind == roundnum:
            count_ind = 0 
            rind += 1
            Vsmp += Vnum
            Tsmp += Tnum
            for i in range(roundnum):
                dat = '\t'.join([str(rind),str(i),str(rule_quality[i]/rind),str(Vmrr[i]/Vsmp),str(Tmrr[i]/Tsmp),str(VH1[i]/Vsmp),str(TH1[i]/Tsmp)])
                #print(f"{rind,i},rule_quality,{dat}")
                if rind ==R:print(dat)
        if rind==R:
            break
    ct = 0
    for i,num in Tnum_dict.items():
        ct += num
        #print(i,num)
    print(rind,ct,"ACT analysis Tnum",Tsmp)
    return Tnum_dict
if __name__ == "__main__":
    filename="rot-cos-path1-inductive-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx300-tr9-noinv-R4+.out"
    curtR = 224
    comTnum1 = analysis(filename,R=curtR)
    filename="rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx375-tr9-noinv-R4+.out"
    comTnum2 = analysis(filename,R=curtR)
    comTnum2 = analysis(filename,comTnum1,R=curtR)