### Google QUEST Q&A Labeling
Improving automated understanding of complex question answer content

In order to run the code install ['A lightweight python library that helps to keep track of numerical experiments'](https://github.com/ex4sperans/mag).<br>
You can find competition data [here](https://www.kaggle.com/c/google-quest-challenge/data).

Example of default bert-base training command from `master` branch:

`run.py --epochs=5 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=1 --batch_size=8 --warmup=300 --lr=1e-5 --bert_model=bert-base-uncased`

Example of BART training command from `bart` branch:

`run.py --epochs=4 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=4 --batch_size=2 --warmup=250 --lr=2e-5 --bert_model=./bart.large`

After you've added a pseudo labels set (we used a 100k subset from [archive](https://archive.org/details/stackexchange)):

`run.py --epochs=4 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=4 --batch_size=2 --warmup=250 --lr=2e-5 --bert_model=./bart.large --pseudo_file ../input/leak-free-pseudo-100k/pseudo-100k-4x-blend-no-leak-fold-{}.csv.gz --split_pseudo --leak_free_pseudo` 

In `monty` branch you can find code for LM pretraining on [stackexchange data](https://archive.org/details/stackexchange)<br>

Read our solution and explanation [here](https://www.kaggle.com/c/google-quest-challenge/discussion/129840).<br>
**To be done**.
