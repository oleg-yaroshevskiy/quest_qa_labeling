#!/bin/bash

(python model1_bert_uncased_code/predict_test.py   \
  --model_dir input/model1_ckpt/                    \
  --data_path input/google-quest-challenge/          \
  --sub_file submissions/model1_submission.csv         \
  > logs/model1_inference.log 2>&1 &)