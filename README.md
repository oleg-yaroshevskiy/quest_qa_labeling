# Google QUEST Q&A Labeling 1st place solution 

Below you can find an outline of how to reproduce our solution for the Google QUEST Q&A Labeling competition. If you run into any trouble with the setup/code or have any questions please contact me at [yury.kashnitsky@gmail.com](mailto:yury.kashnitsky@gmail.com). 

The solution is also described in [this post](https://www.kaggle.com/c/google-quest-challenge/discussion/129840) on Kaggle, the inference part is fully reproduced in [this Kaggle Notebook](https://www.kaggle.com/ddanevskyi/1st-place-solution). 

## Archive contents

[The archive]() contains the following files:

- _to be listed_

## Hardware
- 1 x NVIDIA Quadro P6000
- 2 x NVIDIA 1080 Ti
- 5 x NVIDIA 1080 Ti (only for language model training)

## Software
- Python 3.6.6
- CUDA 10.0.130
- cuDNN 7.5.0
- NVIDIA drivers v. 418.67

Python packages are detailed separately in `requirements.txt`
Apart from pip-installable packages, we use a custom lightweight library called [mag](https://github.com/ex4sperans/mag) to keep track of experiments.


## Reproducing the solution from scratch

This part has the following subparts:

  - Model training
  - Inference
  - Creating final submission

### Model training 

For some of our models, we perform language model finetuning with StackExchange data. Then we run 5-fold cross-validation for 4 models (2 [BERT](https://arxiv.org/abs/1810.04805) ones, one [RoBERTa](https://arxiv.org/abs/1907.11692), and one [BART](https://arxiv.org/abs/1910.13461)) averaging predictions of all 5 model checkpoints for each model type. Finally, blending 4 predictions. In this section, we cover everything related to model training. 

#### 1. Language model finetuning with StackExchange data

Preparing StackExchange data for language model finetuning: `sh bash/train1a_prepare_stackx_data.sh`

Fine-tuning BERT language model with StackExchange data: `sh bash/train1b_train_bert_stackx_lang_model.sh`

#### 2. BERT-base-uncased StackExchange

Training 5 BERT-base models (cross-validation): `sh bash/train2_bert_base_uncased_stackx.sh`. The result (one model checkpoint for each fold) is found in [this Kaggle Dataset](https://www.kaggle.com/dmitriyab/stackx-80-aux-ep-3).

#### 3. BERT-base-cased StackExchange

Training 5 BERT-base models (cross-validation): `sh bash/train3_bert_base_cased_stackx.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/yaroshevskiy/bert-base-pretrained).

#### 4. RoBERTa-base with pseudo-labels

Training 5 RoBERTa-base models (cross-validation): `sh bash/train4_roberta_with_pseudo_labels.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/ddanevskyi/roberta-base-model) and [here](https://www.kaggle.com/dmitriyab/roberta-stackx-base-pl20k) 5 model checkpoints (one per each fold) are stored.


#### 5. BART-large with pseudo-labels

Training 5 BART-large models (cross-validation): `sh bash/train5_bart_with_pseudo_labels.sh`. The result is found in [this Kaggle Dataset](https://www.kaggle.com/yaroshevskiy/bart-large), and [here](https://www.kaggle.com/yaroshevskiy/quest-bart) 5 model checkpoints (one per each fold) are stored.


### Inference
These are the steps to reproduce our final solution (same as our Kaggle Notebook [1st place solution](https://www.kaggle.com/ddanevskyi/1st-place-solution)):

1. Inference with 5 checkpoints of BERT-base-uncased finetuned with StackExchange data: `sh bash/infer1_bert_base_uncased_stackx.sh`
2. Same for the cased BERT model:  `sh bash/infer2_bert_base_cased_stackx.sh`  
3. Inference with with 5 checkpoints of RoBERTa: `sh bash/infer3_roberta.sh`
4. Inference with with 5 checkpoints of BART: `sh bash/infer4_bart.sh`


### Creating final submission
Once inference is done, final steps include blending, and postprocessing model predictions: `sh bash/blending_and_processing.sh` 

