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

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx150-tr9-noinv-R3+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx300-tr9-noinv-R4+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-cos-path1-inductive-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx77-tr9-noinv-R4+.out &


# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > wn-rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx100-tr9-noinv-R3+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > umls-rot-cos-path1-disbuf-grapht_ind-rely_gen_rot2000-l1000x0.1mx35-2-tr9-noinv-R4+.out &


# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > umls-rot-F1-path1-disbuf-grapht_ind-rely_gen_rot-l100x0.1mx50-tr9-noinv-R4+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > umls-rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx35-2-tr9-noinv-R4+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > ilpc-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx400-tr9-noinv-R4.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > ilpcLG-68-rot-cos-path1-disbuf-grapht_ind-rely_gen_one-l1000x0.1mx290-tr9-noinv-R4.out &

CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-cos-path1-grapht_ind--rely_gen_rot-l1000x0.1cx10000mx500-tr9-noinv-R4+.out &

# CUDA_VISIBLE_DEVICES="$@" nohup python -u run_em.py > rot-dim100-cos-path1-grapht_ind--rely_gen_rot-l1000x0.1mx250-tr9-noinv-R3+.out &
