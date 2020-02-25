#!/bin/bash

# competition data
(kaggle competitions download -c google-quest-challenge && \
unzip google-quest-challenge.zip; rm google-quest-challenge.zip; mv *.csv input/google-quest-challenge &)
