#!/bin/bash

python step2_pseudo_labeling/bert-large/infer_pseudo.py                    \
  --experiment=experiments/4-2-5-head_tail-large-1e-05-210-260-500-26-300   \
  --checkpoint=best_model.pth                                                \
  --dataframe=input/sampled_sx_so.csv.gz                                      \
  --output_dir=pseudo-predictions/large/