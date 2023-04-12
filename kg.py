from att import e2wid,eid2wid,args

#print(e2wid('02vqpx8'))
#print(e2wid('05gp3x'))
print(eid2wid(1))
print(eid2wid(10769))

train_sample = "/m/047g8h	/sports/sports_position/players./sports/sports_team_roster/team	/m/01y3c"

parts = train_sample.strip().split("	")
print("head:",e2wid(parts[0].replace("/m/","")))
print("tail:",e2wid(parts[2].replace("/m/","")))

r2id = {}
with open(args.data + "/relation2id.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("	")
        r = parts[0]
        r_id = parts[1]
        r2id[r] = r_id

def gen_predicate_sample(sample_file = args.data + "/train.txt",r_num = 237):
    r_sample = {}    
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("	")
            head = e2wid(parts[0].replace("/m/",""))
            if head==None:continue
            r = parts[1]
            tail = e2wid(parts[2].replace("/m/",""))
            if tail==None:continue
            r_id = r2id[r]
            if r_id not in r_sample:r_sample[r_id] = [(head,tail)]
            elif len(r_sample[r_id])<3:r_sample[r_id].append((head,tail))
    for k in r_sample:
        print("Predicate",k,"samples:",r_sample[k])
    return r_sample

#gen_predicate_sample()
if __name__ == "__main__":
    print("kg util...")
    #gen_predicate_sample()