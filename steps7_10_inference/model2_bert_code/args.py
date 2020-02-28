import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="../input/")
parser.add_argument("--checkpoints", type=str, default="../input/")
parser.add_argument("--pseudo_file", type=str)
parser.add_argument("--split_pseudo", action="store_true", default=False)
parser.add_argument("--leak_free_pseudo", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--folds", type=int, default=5)

parser.add_argument("--label", type=str, default="qa")
parser.add_argument("--bert_model", type=str, default="bert-large-uncased")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_accumulation", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup", type=int, default=200)

# loss
parser.add_argument("--num_classes", type=int, default=30)
parser.add_argument("--workers", type=int, default=8)

# tokenization
parser.add_argument("--max_sequence_length", type=int, default=290)
parser.add_argument("--max_title_length", type=int, default=30)
parser.add_argument("--max_question_length", type=int, default=128)
parser.add_argument("--max_answer_length", type=int, default=128)
parser.add_argument("--head_tail", type=str, default="True")

# infer
parser.add_argument("--sub_file", type=str, default="submission.csv")

args = parser.parse_args()

for arg in ["head_tail"]:
    args.__dict__[arg] = args.__dict__[arg] == "True"

for arg in ["lr"]:
    args.__dict__[arg] = float(args.__dict__[arg])
print("Initial arguments", args)

args.__dict__["input_columns"] = ["question_title", "question_body", "answer"]
args.__dict__["target_columns"] = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]
