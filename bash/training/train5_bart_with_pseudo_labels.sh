#!/bin/bash

python model4_bart_code/run.py \
--data_path=data \
--epochs=4 \
--max_sequence_length=500 \
--max_title_length=26 \
--max_question_length=260 \
--max_answer_length=210 \
--batch_accumulation=8 \
--batch_size=1 \
--warmup=250 \
--lr=2e-5 \
--bert_model=bart.large \
--pseudo_file=pseudo-predictions/pseudo-100k-3x-blend-no-leak/fold-{}.csv.gz \
--split_pseudo \
--leak_free_pseudo \
--label=bart