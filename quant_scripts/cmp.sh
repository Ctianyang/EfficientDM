#!/bin/bash
GPUS="0,2" 
OUT_DIR="./generated"

script=./quant_scripts
num_gpus=$(($(echo $GPUS | grep -o "," | wc -l) + 1))

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$num_gpus \
    --master_port 6000 \
    --log_dir $OUT_DIR \
   ${script}/cmp_qalra_fp.py \
    --num_samples 500 \
    --batch_size 16 \
    --out_dir $OUT_DIR \
    # --resume  # 取消注释以启用恢复模式