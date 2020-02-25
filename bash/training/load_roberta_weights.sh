#!/bin/bash

mkdir input/roberta-base/; \
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json -O  input/roberta-base/config.json;\
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json -O  input/roberta-base/vocab.json;\
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt -O  input/roberta-base/merges.json;\
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin -O  input/roberta-base/pytorch_model.bin