#!/bin/bash

# scraping StackExchange
python step1_lm_finetuning/data_preparation/scrape_stack_exchange.py

# processing results
python step1_lm_finetuning/data_preparation/clean_stack_exchange_qa.py