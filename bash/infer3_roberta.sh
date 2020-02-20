python ../input/roberta-base-code/infer.py                    \
  --experiment $ROBERTA_EXPERIMENT_DIR                         \
  --checkpoint=best_model.pth                                   \
  --bert_model=/kaggle/input/roberta-base-model                     \
  --dataframe=/kaggle/input/google-quest-challenge/test.csv   \
  --output_dir=roberta-base-output