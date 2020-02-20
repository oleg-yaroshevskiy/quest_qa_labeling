#!/bin/bash

python pseudo-models/bert-base/infer_pseudo.py \
--experiment=experiments/1-8-5-head_tail-qa-1e-05-210-260-500-26-300 \
--checkpoint=best_model.pth \
--dataframe=data/sampled_sx_so.csv.gz \
--output_dir=pseudo-predictions/base/