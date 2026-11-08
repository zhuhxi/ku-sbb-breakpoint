#!/bin/bash
# usage: probe/gen_zs.sh NAME 'python dict updates' SEEDBASE GPU SIMDIR  -> zero-shot AUC after mtw + 1 epoch on 600k fresh fakes
NAME=$1; UPD=$2; SB=$3; G=$4; D=$5; mkdir -p $D; [ "$(ls $D | wc -l)" -le 12 ] || { echo "$D has extra files"; exit 1; }
python -c "import json; b=json.load(open('outputs/factory/'+__import__('os').environ.get('BASEGEN','ku_gen_nuisld0')+'.json')); b.update($UPD); json.dump(b,open('outputs/factory/ku_gen_h_$NAME.json','w'))"
for i in $(seq 0 11); do timeout 60 python probe/sim_tensor.py 50000 outputs/factory/ku_gen_h_$NAME.json $D/sim_$i.npz $((SB + i*50000)) > /dev/null 2>&1; done
SIMDIR=$D NFILE=12 MT=1 INIT=outputs/arms/seqpre_mtw_c64.pt LRMAX=5e-4 OUT=outputs/arms/seqpre_h_${NAME}_mt_c64.pt CUDA_VISIBLE_DEVICES=$G timeout 200 ~/venvs/tabpfn/bin/python -W ignore probe/seq_pretrain.py 1 2>&1 | tail -n 1 | sed "s/^/$NAME: /"
