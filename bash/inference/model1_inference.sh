#!/bin/bash

python steps7_10_inference/model1_bert_code/predict_test.py      \
  --model_dir input/model1_ckpt/                                  \
  --data_path input/google-quest-challenge/                        \
  --sub_file submissions/model1_submission.csv                      \
  > logs/model1_inference.log 2>&1
