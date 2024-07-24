#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import inspect
import codecs
import argparse
import tempfile
import warnings
from collections import Counter
from multiprocessing import cpu_count

#hack to get imports working if running this as a script, or within a package
if __name__ == '__main__':
    import learn_bpe
    import apply_bpe
else:
    from . import learn_bpe
    from . import apply_bpe

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('learn-joint-bpe-and-vocab',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('rb'), required=True, nargs = '+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('wb'), required=True,
        metavar='PATH',
        help="Output file for BPE codes.")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--byte', '-b', action="store_true",
        help="byte-level BPE.")
    parser.add_argument(
        '--separator', type=bytes, default=b'@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s')")
    parser.add_argument(
        '--write-vocabulary', type=argparse.FileType('wb'), required=True, nargs = '+', default=None,
        metavar='PATH', dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser

def learn_joint_bpe_and_vocab(args):

    if args.vocab and len(args.input) != len(args.vocab):
        sys.stderr.write('Error: number of input files and vocabulary files must match\n')
        sys.exit(1)

    if args.byte:
        # read/write files as byte streams
        args.input = [codecs.open(f.name, 'rb') for f in args.input]
        args.vocab = [codecs.open(f.name, 'wb') for f in args.vocab]
    else:
        # read/write files as UTF-8
        args.input = [codecs.open(f.name, encoding='UTF-8') for f in args.input]
        args.vocab = [codecs.open(f.name, 'w', encoding='UTF-8') for f in args.vocab]

    args.separator = args.separator.decode('UTF-8') if not args.byte else args.separator

    # get combined vocabulary of all input texts
    full_vocab = Counter()
    for f in args.input:
        full_vocab += learn_bpe.get_vocabulary(f, num_workers=args.num_workers, is_bytes=args.byte)
        f.seek(0)

    if args.byte:
        vocab_list = [key + b' ' + str(freq).encode('UTF-8') for (key, freq) in full_vocab.items()]
    else:
        vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary

    if args.byte:
        with open(args.output.name, 'wb') as output:
            learn_bpe.learn_bpe(vocab_list, output, args.symbols, args.min_frequency, args.verbose, is_dict=True, is_bytes=args.byte, total_symbols=args.total_symbols)

        with open(args.output.name, 'rb') as codes:
            bpe = apply_bpe.BPE(codes, separator=args.separator, is_bytes=args.byte)
    else:
        with codecs.open(args.output.name, 'w', encoding='UTF-8') as output:
            learn_bpe.learn_bpe(vocab_list, output, args.symbols, args.min_frequency, args.verbose, is_dict=True, is_bytes=args.byte, total_symbols=args.total_symbols)

        with codecs.open(args.output.name, encoding='UTF-8') as codes:
            bpe = apply_bpe.BPE(codes, separator=args.separator, is_bytes=args.byte)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(args.input, args.vocab):

        # read/write files as UTF-8
        if not args.byte:
            train_file = codecs.open(train_file.name, encoding='utf-8')
            vocab_file = codecs.open(vocab_file.name, 'w', encoding='utf-8')

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        if args.byte:
            tmpout = open(tmp.name, 'wb')
        else:
            tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        bpe.process_lines(train_file.name, tmpout, num_workers=args.num_workers)

        tmpout.close()

        if args.byte:
            tmpin = open(tmp.name, 'rb')
        else:
            tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab = learn_bpe.get_vocabulary(tmpin, num_workers=args.num_workers, is_bytes=args.byte)
        tmpin.close()
        os.remove(tmp.name)

        for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            if args.byte:
                vocab_file.write(key + b" " + str(freq).encode('utf-8') + b"\n")
            else:
                vocab_file.write("{0} {1}\n".format(key, freq))
        train_file.close()
        vocab_file.close()


if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        print("Python 2 is deprecated. Use Python 3")
        sys.exit(1)

    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()

    if args.num_workers <= 0:
        args.num_workers = cpu_count()

    assert(len(args.input) == len(args.vocab))

    learn_joint_bpe_and_vocab(args)
