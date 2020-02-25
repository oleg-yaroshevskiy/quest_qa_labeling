#!/bin/bash

# train three models
sh bash/pseudo/train_base.sh                         && \
sh bash/pseudo/train_base_pretrained.sh              && \
sh bash/pseudo/train_large.sh

# create pseudo-labels with all models
sh bash/pseudo/create_pseudo_base.sh                 && \
sh bash/pseudo/create_pseudo_base_pretrained.sh      && \
sh bash/pseudo/create_pseudo_large.sh

# blend pseudo-labels
python step2_pseudo_labeling/blend_pseudo.py
