#!/bin/bash

python pseudo-models/bert-base-pretrained/run.py \
--epochs=3 \
--max_sequence_length=500 \
--max_title_length=26 \
--max_question_length=260 \
--max_answer_length=210 \
--data_path=data \
--batch_accumulation=1 \
--batch_size=8 \
--warmup=100 \
--lr=1e-5 \
--bert_model=data/stackx-base-cased \
--label=pretrained