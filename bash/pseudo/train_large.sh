#!/bin/bash

python pseudo-models/bert-large/run.py \
--epochs=5 \
--max_sequence_length=500 \
--max_title_length=26 \
--max_question_length=260 \
--max_answer_length=210 \
--data_path=data \
--batch_accumulation=4 \
--batch_size=2 \
--warmup=300 \
--lr=1e-5 \
--bert_model=bert-large-uncased \
--label=large
