#!/bin/bash

python steps7_10_inference/model4_bart_code/run.py  \
  --sub_file=submissions/model4_bart_large_pred.csv  \
  --data_path=input/google-quest-challenge/           \
  --max_sequence_length=500                            \
  --max_title_length=26                                 \
  --max_question_length=260                              \
  --max_answer_length=210                                 \
  --batch_size=4                                           \
  --bert_model=input/model4_ckpt/bart.large/                \
  > logs/model4_inference.log 2>&1
