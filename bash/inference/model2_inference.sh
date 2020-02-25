#!/bin/bash

python steps7_10_inference/model2_bert_code/run.py       \
  --sub_file=submissions/model2_bert_base_cased_pred.csv  \
  --data_path=input/google-quest-challenge/                \
  --max_sequence_length=500                                 \
  --max_title_length=26                                      \
  --max_question_length=260                                   \
  --max_answer_length=210                                      \
  --batch_size=8                                                \
  --bert_model=input/model2_ckpt/                                \
  --checkpoints=input/model2_ckpt/bert-base-pseudo-noleak-random  \
  > logs/model2_inference.log 2>&1
