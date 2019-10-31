# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for RoBERTa."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open

from .tokenization_utils import PreTrainedTokenizer
import youtokentome as yttm

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab_30000.bpe',
    'merges_file': 'dict.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
}

class RubertaTokenizer(PreTrainedTokenizer):
    """
    RuBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, errors='replace', bos_token="<s>", eos_token="</s>", sep_token="</s>",
                 cls_token="<s>", unk_token="<unk>", pad_token='<pad>', mask_token='<mask>', **kwargs):
        super(RubertaTokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.bpe = yttm.BPE(model=vocab_file)
        self.init_converter(merges_file)


    def init_converter(self, filename):
        self.bpe2dict = {0:0, 1:1, 2:2, 3:3}
        self.dict2bpe = {0:0, 1:1, 2:2, 3:3}
        with open(filename) as f:
            for i, line in enumerate(f):
                try:
                    idx = int(line.split(' ')[0])
                except ValueError:
                    continue
                self.bpe2dict[idx] = i + 4
                self.dict2bpe[i + 4] = idx


    def add_special_tokens_single_sequence(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        A RoBERTa sequence has the following format: <s> X </s>
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s> B </s>
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def _tokenize(self, text):
        return self.bpe.encode(text, output_type=yttm.OutputType.SUBWORD)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        bpe_idx = self.bpe.subword_to_id(token)
        return self.bpe2dict.get(bpe_idx, 3)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        index = self.dict2bpe(index, 3)
        return self.bpe.id_to_subword(index)

    def convert_tokens_to_string(self, tokens):
        return self.bpe.decode([self.bpe.subword_to_id(token) for token in tokens])[0]

    def convert_tokens_to_ids_(self, tokens):
        res = []
        for token in tokens:
            res += self.bpe.encode(token)
        return res

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
        os.system('cp {} {}'.format(self.vocab_file, vocab_file))
        merges_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])
        os.system('cp {} {}'.format(self.merges_file, merges_file))
        return (vocab_file, )

