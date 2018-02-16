# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
This script learns BPE jointly on a concatenation of a list of texts (typically the source and target side of a parallel corpus,
applies the learned operation to each and (optionally) returns the resulting vocabulary of each text.
The vocabulary can be used in apply_bpe.py to avoid producing symbols that are rare or OOV in a training text.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import sys
import os
import codecs
import tempfile
from collections import Counter

from subword_nmt.learn_bpe import learn_bpe, get_vocabulary
from subword_nmt.apply_bpe import BPE


def learn_joint_bpe_and_vocab(args):
    if args.vocab and len(args.input) != len(args.vocab):
        sys.stderr.write('Error: number of input files and vocabulary files must match\n')
        sys.exit(1)

    # read/write files as UTF-8
    args.input = [codecs.open(f.name, encoding='UTF-8') for f in args.input]
    args.vocab = [codecs.open(f.name, 'w', encoding='UTF-8') for f in args.vocab]

    # get combined vocabulary of all input texts
    full_vocab = Counter()
    for f in args.input:
        full_vocab += get_vocabulary(f)
        f.seek(0)

    vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary
    with codecs.open(args.output.name, 'w', encoding='UTF-8') as output:
        learn_bpe(vocab_list, output, args.symbols, args.min_frequency, args.verbose, is_dict=True)

    with codecs.open(args.output.name, encoding='UTF-8') as codes:
        bpe = BPE(codes, separator=args.separator)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(args.input, args.vocab):

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        for line in train_file:
            tmpout.write(bpe.segment(line).strip())
            tmpout.write('\n')

        tmpout.close()
        tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab = get_vocabulary(tmpin)
        tmpin.close()
        os.remove(tmp.name)

        for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write("{0} {1}\n".format(key, freq))
        vocab_file.close()
