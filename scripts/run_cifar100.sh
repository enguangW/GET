#!/bin/bash

set -e
set -x

cuda_id=0
TES_exp_name="TES2_CIFAR100"
GET_exp_name="GET2_CIFAR100_mw_4_lc_1"

exp_id="${TES_exp_name}_${GET_exp_name}_$(date +%Y%m%d_%H%M%S)"


python s1_TES.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --cuda_dev $cuda_id \
    --exp_name $TES_exp_name \
    --exp_id $exp_id 


python s2_GET.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --cuda_dev $cuda_id \
    --exp_name $GET_exp_name\
    --exp_id $exp_id \



