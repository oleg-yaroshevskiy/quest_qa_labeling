#!/bin/bash

toy=${1:-False}

if [ $toy = 'toy' ]; then
    epochs=1
else
    epochs=5
fi

python step5_model3_roberta_code/run.py \
    --epochs=5 \
    --max_sequence_length=500 \
    --max_title_length=26 \
    --max_question_length=260 \
    --max_answer_length=210 \
    --data_path=input/google-quest-challenge \
    --batch_accumulation=2 \
    --batch_size=4 \
    --warmup=200 \
    --lr=1e-5 \
    --bert_model=input/roberta-base \
    --label=pseudonoleakrandom100k \
    --pseudo_file pseudo-predictions/pseudo-100k-3x-blend-no-leak/fold-{}.csv.gz \
    --n_pseudo=20000 \
    --model_type=roberta \
    --toy=$toy

