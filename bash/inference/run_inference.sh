#!/bin/bash

#echo "Downloading all model checkpoints"                                     && \
#sh bash/download_all_model_ckpts_for_inference.sh                            && \

echo "Inference with the 1st model (BERT-base-cased)"                         && \
sh bash/inference/model1_inference.sh                                         && \

echo "Inference with the 2nd model (BERT-base-cased with pseudo-labels)"      && \
sh bash/inference/model2_inference.sh                                         && \

echo "Inference with the 3rd model (RoBERTa with pseudo-labels)"              && \
sh bash/inference/model3_inference.sh                                         && \

echo "Inference with the 4th model (BART with pseudo-labels)"                 && \
sh bash/inference/model4_inference.sh                                         && \

echo "Blending and postprocessing"                                            && \
sh bash/blending_n_postprocessing.sh                                             \

