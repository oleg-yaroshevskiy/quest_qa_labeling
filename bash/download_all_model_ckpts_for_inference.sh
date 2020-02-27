#!/bin/bash

# model 1
(kaggle datasets download -d kashnitsky/google-qa-quest-labeling-bibimorph-model-1-5-folds > /dev/null 2>&1 && \
unzip google-qa-quest-labeling-bibimorph-model-1-5-folds.zip -d input/model1_ckpt \
&& rm google-qa-quest-labeling-bibimorph-model-1-5-folds.zip &)

# model 2
(kaggle datasets download -d yaroshevskiy/bert-base-pretrained > /dev/null 2>&1                            && \
unzip bert-base-pretrained.zip -d input/ && bert-base-pretrained.zip                                       && \
mv input/stackx-base-cased input/model2_ckpt &)

(kaggle datasets download -d  ddanevskyi/bert-base-pseudo-noleak-random > /dev/null 2>&1 && \
unzip bert-base-pseudo-noleak-random.zip -d input/model2_ckpt/bert-base-pseudo-noleak-random && \
rm bert-base-pseudo-noleak-random.zip)

# model 3
(kaggle datasets download -d kashnitsky/google-qa-quest-labeling-bibimorph-model-3-roberta > /dev/null 2>&1 && \
mkdir input/model3_ckpt/ && \
unzip google-qa-quest-labeling-bibimorph-model-3-roberta.zip -d input/model3_ckpt/folds                     && \
rm google-qa-quest-labeling-bibimorph-model-3-roberta.zip &)

(kaggle datasets download -d ddanevskyi/roberta-base-model > /dev/null 2>&1                                 && \
unzip roberta-base-model.zip -d input/model3_ckpt/roberta-base-model                                        && \
rm roberta-base-model.zip  &)

# model 4
(kaggle datasets download -d yaroshevskiy/quest-bart > /dev/null 2>&1                                       && \
unzip quest-bart.zip -d input/model4_ckpt && rm quest-bart.zip                                              && \
kaggle datasets download -d yaroshevskiy/bart-large > /dev/null 2>&1                                        && \
unzip bart-large.zip -d input/model4_ckpt && rm bart-large.zip        &)

