#!/bin/bash

# install mag,
# a custom lightweight library to keep track of experiments
git clone https://github.com/ex4sperans/mag.git
cd mag; python setup.py install; cd ../;

# install hacked version of fairseq
# Sorry, this is very-very hacky and ugly but works
export PATH_TO_GPT2BPE=packages/gpt2bpe; \
export PATH_TO_BART_MODEL=input/model4_ckpt/bart.large/; \
cd packages/fairseq-hacked/; python setup.py develop; cd ../..
