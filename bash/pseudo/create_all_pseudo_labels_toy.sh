#!/bin/bash

# train three models
sh bash/pseudo/train_base.sh toy                         && \
sh bash/pseudo/train_base_pretrained.sh toy              && \
sh bash/pseudo/train_large.sh toy

# create pseudo-labels with all models
sh bash/pseudo/create_pseudo_base.sh toy                 && \
sh bash/pseudo/create_pseudo_base_pretrained.sh toy      && \
sh bash/pseudo/create_pseudo_large.sh toy

# blend pseudo-labels
python step2_pseudo_labeling/blend_pseudo.py toy
