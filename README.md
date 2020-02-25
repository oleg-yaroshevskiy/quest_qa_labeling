# Google QUEST Q&A Labeling 1st place solution 

Below you can find an outline of how to reproduce our solution for the Google QUEST Q&A Labeling competition. If you run into any trouble with the setup/code or have any questions please contact me at [yury.kashnitsky@gmail.com](mailto:yury.kashnitsky@gmail.com). 

The solution is also described in [this post](https://www.kaggle.com/c/google-quest-challenge/discussion/129840) on Kaggle, the inference part is fully reproduced in [this Kaggle Notebook](https://www.kaggle.com/ddanevskyi/1st-place-solution). 

## Archive contents

[The archive]() contains the following files:

- ...

## Hardware
- 1 x NVIDIA Quadro P6000
- 2 x NVIDIA 1080 Ti
- 5 x NVIDIA 1080 Ti (only for language model training)

## Software and Conda environment
- Conda 4.7.10 with Python 3.6.6
- CUDA 10.0.130
- cuDNN 7.5.0
- NVIDIA drivers v. 418.67

We run all experiments in a Conda environment and provide a full list of required packages `requirements_full.txt`  To do the same, run:

 - `conda create -n qa_quest_env python=3.6.6`
 - `conda activate qa_quest_env ` 
 - `pip install -r requirements_full.txt`

However, there are a lot of packages listed there which are not much relevant to this project. A minimal list of requirements is specified in `requirements_minimal.txt` which was formed with the `pipreqs` utility. This handy utility crawls project source code and lists all required packages. So you can run `pip install -r requirements_minimal.txt` and then install missing packages on the go (if any).

Apart from pip-installable packages, we use a custom lightweight library called [mag](https://github.com/ex4sperans/mag) to keep track of experiments. Also our installation of the `fairseq` library is a bit different. These two can be installed by running `sh bash/setup.sh`


## Model training 

For some of our models, we perform language model finetuning with StackExchange data. Then we run 5-fold cross-validation for 4 models (2 [BERT](https://arxiv.org/abs/1810.04805) ones, one [RoBERTa](https://arxiv.org/abs/1907.11692), and one [BART](https://arxiv.org/abs/1910.13461)) averaging predictions of all 5 model checkpoints for each model type. Finally, blending 4 predictions. In this section, we cover everything related to model training:

 1. Language model finetuning with StackExchange data
 1. Generating pseudo-labels
 1. Training BERT-base-cased pretrained with StackExchange
 1. Training BERT-base-cased pretrained with StackExchange + pseudo-labels
 1. Training RoBERTa-base with pseudo-labels
 1. Training BART-large with pseudo-labels


### 1. Language model finetuning with StackExchange data

#### 1a. Scraping and processing StackExchange questions and answers

For this purpose, we download and process StackExchange dumps, to reproduce this steps from scratch for a small subsample, run (the same is done by running `sh bash/training/train1a_prepare_stackx_data.sh`)

- `python step1_lm_finetuning/data_preparation/scrape_stack_exchange.py`
- `python step1_lm_finetuning/data_preparation/clean_stack_exchange_qa.py`

See comments in these files. Also, you can find 2 corresponding Jupyter notebooks in the same folder, just for demonstration. 

Output is written to the `input/qa_stackexchange_cleaned.tsv` file.

This files is needed for the next step.

#### 1b. Fine-tuning BERT language model with StackExchange data:

`sh bash/training/train1b_train_bert_stackx_lang_model.sh` - this runs fine-tuning BERT language model with StackExchange data from the previous step (`input/qa_stackexchange_cleaned.tsv`). 

This script writes model checkpoints and training logs to `input/stackx-large-cased`, the following two BERT models use this checkpoint. 

### 2. Generating pseudo-labels
TODO

### 3. BERT-base-cased pretrained with StackExchange

TODO. Training 5 BERT-base models (cross-validation): `sh bash/training/train2_bert_base_uncased_stackx.sh`. The result (one model checkpoint for each fold) is found in [this Kaggle Dataset](https://www.kaggle.com/dmitriyab/stackx-80-aux-ep-3).

### 4. BERT-base-cased pretrained with StackExchange + pseudo-labels

TODO. Training 5 BERT-base models (cross-validation): `sh bash/training/train3_bert_base_cased_stackx.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/yaroshevskiy/bert-base-pretrained).

### 5. RoBERTa-base with pseudo-labels

TODO. Training 5 RoBERTa-base models (cross-validation): `sh bash/training/train4_roberta_with_pseudo_labels.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/ddanevskyi/roberta-base-model) and [here](https://www.kaggle.com/dmitriyab/roberta-stackx-base-pl20k) 5 model checkpoints (one per each fold) are stored.


### 6. BART-large with pseudo-labels

TODO. Training 5 BART-large models (cross-validation): `sh bash/training/train5_bart_with_pseudo_labels.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/yaroshevskiy/bart-large), and [here](https://www.kaggle.com/yaroshevskiy/quest-bart) 5 model checkpoints (one per each fold) are stored.


## Inference
These are the steps to reproduce our final solution (same as our Kaggle Notebook [1st place solution](https://www.kaggle.com/ddanevskyi/1st-place-solution)). You can just run `sh bash/inference/run_inference.sh` (execution log is found in `logs/all_inference.log`, takes ~10 minutes) to run the whole pipeline (from data loading to forming a submission file). Here is a breakdown of all steps:

1. Make sure you've got a fresh [Kaggle API token](https://www.kaggle.com/docs/api) and run `sh bash/download_all_model_ckpts_for_inference.sh`. This will download all models needed for inference (about 18 Gb, might take from several minutes to more that an hour depending on Internet speed):
 - BERT checkpoints from [this Dataset](https://www.kaggle.com/kashnitsky/google-qa-quest-labeling-bibimorph-model-1-5-folds) (the result of running steps 1, 3 above)
 - BERT checkpoints from [this Dataset](https://www.kaggle.com/yaroshevskiy/bert-base-pretrained) (the result of running steps 2, 4 above)
 - RoBERTa checkpoints from [this Dataset](https://www.kaggle.com/kashnitsky/google-qa-quest-labeling-bibimorph-model-3-roberta) (the result of running steps 1, 2, 5 above)
 - BART checkpoints from [this Dataset](https://www.kaggle.com/yaroshevskiy/quest-bart) (the result of running steps 2, 6 above)

2. Inference with 5 checkpoints of BERT-base-cased finetuned with StackExchange data: `sh bash/inference/model1_inference.sh`
3. Same for the BERT model with pseudo-labels:  `sh bash/inference/model2_inference.sh`
4. Inference with with 5 checkpoints of RoBERTa finetuned with StackExchange data, with pseudo-labels: `sh bash/inference/model3_inference.sh`
5. Inference with with 5 checkpoints of BART with pseudo-labels: `sh bash/inference/model4_inference.sh`
6. Once inference is done, final steps include blending, and postprocessing model predictions: `sh bash/blending_n_postprocessing.sh` 

Final submission `submissions/submission.csv` scores 0.46893 on the public competition [leaderboard](https://www.kaggle.com/c/google-quest-challenge/leaderboard). To actually submit a CSV file, you can check [this Kaggle Notebook](https://www.kaggle.com/kashnitsky/google-quest-q-a-submit-from-a-csv-file).

