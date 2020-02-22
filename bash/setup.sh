#!/bin/bash

# Create a fresh virtual environment
virtualenv -p python3 qa_quest_venv
source qa_quest_venv/bin/activate

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Additionally, install mag
git clone https://github.com/ex4sperans/mag.git
cd mag; python setup.py install; cd ../;

# Additionally, install hacked version of fairseq
# Sorry, this is very-very hacky and ugly but works
export PATH_TO_GPT2BPE=packages/gpt2bpe; \
export PATH_TO_BART_MODEL=input/model4_ckpt/bart.large/; \
cd packages/fairseq/; python setup.py develop; cd ../..