#!/bin/bash

# some mag setup to go on with the Experiment
ROBERTA_EXPERIMENT_DIR=2-4-roberta-base-saved-5-head_tail-roberta-stackx-base-v2-pl1kksample20k-1e-05-210-260-500-26-roberta-200
OUTPUT_DIR=submissions/model3_roberta-base-output
 
cp -r input/model3_ckpt/folds/* experiments/$ROBERTA_EXPERIMENT_DIR

python steps7_10_inference/model3_roberta_code/infer.py    \
  --experiment=$ROBERTA_EXPERIMENT_DIR                      \
  --checkpoint=best_model.pth                                \
  --bert_model=input/model3_ckpt/roberta-base-model/          \
  --dataframe=input/google-quest-challenge/test.csv            \
  --output_dir=$OUTPUT_DIR                                      \
  > logs/model3_inference.log 2>&1
