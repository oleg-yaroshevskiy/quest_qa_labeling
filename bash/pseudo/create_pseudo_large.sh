#!/bin/bash

python pseudo-models/bert-large/infer_pseudo.py \
--experiment=experiments/4-2-5-head_tail-large-1e-05-210-260-500-26-300 \
--checkpoint=best_model.pth \
--dataframe=data/sampled_sx_so.csv.gz \
--output_dir=pseudo-predictions/large/