#!/bin/bash
time1 = $(date +%Y-%m-%d-%H%M)

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rlloss-recall-best-rely_gen_one+-l1000mx1000-tr10-noinv-R4++-"$time1".out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-recall-best+-rely_gen_one-l1000mx200-tr9-noinv-R4+.out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rlloss-recall-best+-204-rely_gen_one+-l1000mx200-tr9-noinv-R4++-"$time1".out &


#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-1-cos-rely_gen_one+-l1000mx200-tr9-noinv-R4++.out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-F2-rely_gen_one-l1000mx200-tr9-noinv-R4+.out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-cos-bestv-rely_gen_one-l1000mx200-tr9-noinv-R4+.out &
#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-F1-bestv-rely_gen_one-l10mx200-tr9-noinv-R4+.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrsloss-cos-bestv-rely_gen_one-l1000-0.1mx200-tr9-noinv-R4+.out &


#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > ind22-rltrs-cos-bestv-grapht_ind2-rely_gen_one-l1000x0.1mx200-tr9-noinv-R4+.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rltrs-cos-bestv-graphtest-93-rely_gen_one-l1000x0.1mx100-tr9-noinv-R4+.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > cosF1-2lnj-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx50-tr9-noinv-R4+.out &

#CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > cos-path5--disbuf-grapht_ind-rely_gen_one-l1000x0.1mx200-tr9-noinv-R4+.out &

CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx325-tr9-noinv-R4+.out &