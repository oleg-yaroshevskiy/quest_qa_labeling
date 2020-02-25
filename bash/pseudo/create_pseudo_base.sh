#!/bin/bash

python step2_pseudo_labeling/bert-base/infer_pseudo.py                  \
  --experiment=experiments/1-8-5-head_tail-qa-1e-05-210-260-500-26-300   \
  --checkpoint=best_model.pth                                             \
  --dataframe=input/qa_stackexchange_cleaned.tsv                           \
  --output_dir=pseudo-predictions/base/