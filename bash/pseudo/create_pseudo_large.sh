#!/bin/bash

toy=${1:-False}

if [ $toy = 'toy' ]; then
    dataframe=input/qa_stackexchange_cleaned_toy.csv
else
    dataframe=input/qa_stackexchange_cleaned.csv
fi

python step2_pseudo_labeling/bert-large/infer_pseudo.py                    \
  --experiment=experiments/4-2-5-head_tail-large-1e-05-210-260-500-26-300   \
  --checkpoint=best_model.pth                                                \
  --dataframe=$dataframe			                                          \
  --output_dir=pseudo-predictions/large/
