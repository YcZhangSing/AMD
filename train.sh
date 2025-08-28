#!/bin/bash

random_seed=12 
export PYTHONHASHSEED=$random_seed
export CUDA_VISIBLE_DEVICES=0,1,2,3  

MODEL_PTH='AMD/models'
TRAIN_JS='path/to/train.json'
VAL_JS='path/to/val.json'

python scripts/train.py \
    --AMD-init-pth "$MODEL_PTH" \
    --dataset DGM4 \
    --batch-size 5 \
    --epochs 13 \
    --lr 1e-6 \
    --eval-steps 2000 \
    --run-name "amd_train_run_$(date +%Y%m%d_%H%M)" \
    --max-val-item-count 2000 \
    --regular-weight 2000 \
    --train-js "$TRAIN_JS" \
    --val-js "$VAL_JS" \
    --train-domain "NYT"\
    --seed $random_seed

echo "Training completed at: $(date)"
