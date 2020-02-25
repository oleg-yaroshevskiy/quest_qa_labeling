#!/bin/bash

# competition data
(kaggle competitions download -c google-quest-challenge && \
unzip train.csv.zip; rm train.csv.zip; mv *.csv input/google-quest-challenge &)
