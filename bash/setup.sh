#!/bin/bash

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Additionally, install mag
TODO

# Additionally, install hacked version of fairseq
# Sorry, this is very-very hacky and ugly, but works
export PATH_TO_GPT2BPE=packages/gpt2bpe; \
export PATH_TO_BART_MODEL=input/model4_ckpt/bart.large/; \
cd packages/fairseq/; python setup.py develop; cd ../..
