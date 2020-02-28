import numpy as np
from typing import List
from transformers import BertTokenizer


class BertRandomTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        lower_case_prob=1.0,
        basic_tokenize_prob=1.0,
        piece_split_prob=0.0,
        recursive=False,
        **kwargs
    ):
        super(BertRandomTokenizer, self).__init__(
            vocab_file,
            do_lower_case=lower_case_prob > 0,
            do_basic_tokenize=True,
            **kwargs
        )
        self.lower_case_prob = lower_case_prob
        self.basic_tokenize_prob = basic_tokenize_prob
        self.piece_split_prob = piece_split_prob
        self.recursive = recursive

    def _split_word_piece(self, token: str) -> List[str]:
        if token in self.all_special_tokens:
            return [token]

        token_id = self.wordpiece_tokenizer.vocab.pop(token)
        pieces = self.wordpiece_tokenizer.tokenize(token)
        self.wordpiece_tokenizer.vocab[token] = token_id

        if self.unk_token not in pieces and len(pieces) > 0:
            return pieces
        else:
            return [token]

    def _tokenize(self, text):
        self.basic_tokenizer.do_lower_case = (
            np.random.uniform(0, 1) <= self.lower_case_prob
        )
        self.do_basic_tokenize = np.random.uniform(0, 1) <= self.basic_tokenize_prob

        def unroll(token_list):
            tokens = []
            for item in token_list:
                if not isinstance(item, list):
                    tokens.append(item)
                else:
                    tokens += item
            return tokens

        tokens = super(BertRandomTokenizer, self)._tokenize(text)

        current_pos = 0
        while current_pos < len(tokens):
            if np.random.uniform(0, 1) <= self.piece_split_prob:
                tokens[current_pos] = self._split_word_piece(tokens[current_pos])
                if self.recursive:
                    tokens = unroll(tokens)
            current_pos += 1

        tokens = unroll(tokens)
        return tokens
