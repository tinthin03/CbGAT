#!/bin/bash
time1 = $(date)
UDA_VISIBLE_DEVICES="$@" python -u run_em.py > trainEM${time1}.out 
