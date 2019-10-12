# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe

import youtokentome as yttm

DEFAULT_VOCAB_BPE = '../output/vocab.bpe'


@register_bpe('yttm')
class YTTMBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--vocab-bpe', type=str,
                            default=DEFAULT_VOCAB_BPE,
                            help='path to vocab.bpe')
        # fmt: on

    def __init__(self, args):
        vocab_bpe = file_utils.cached_path(
            getattr(args, 'vocab_bpe', DEFAULT_VOCAB_BPE)
        )
        self.bpe = yttm.BPE(model=vocab_bpe)

    def encode(self, x: str) -> str:
        return ' '.join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(map(int, x.split()))

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith('▁')


def get_encoder(args):
    return YTTMBPE(args)
