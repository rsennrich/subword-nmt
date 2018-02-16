import io
import sys
import codecs
import argparse

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE, read_vocabulary
from subword_nmt.get_vocab import get_vocab
from subword_nmt.segment_char_ngrams import segment_char_ngrams
from subword_nmt.learn_joint_bpe_and_vocab import learn_joint_bpe_and_vocab


# hack for python2/3 compatibility
argparse.open = io.open


def create_learn_bpe_parser(subparsers):
    parser = subparsers.add_parser('learn-bpe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser


def create_apply_bpe_parser(subparsers):
    parser = subparsers.add_parser('apply-bpe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. The strings provided in glossaries will not be affected"+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords")

    return parser


def create_get_vocab_parser(subparsers):
    parser = subparsers.add_parser('get-vocab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generates vocabulary")

    parser.add_argument(
        '--train_file', type=str, default=None,
        metavar="PATH",
        help="File where to save vocab.")

    parser.add_argument(
        '--vocab_file', type=str, default=None,
        metavar="PATH",
        help="File where to save vocab.")

    return parser


def create_segment_char_ngrams_parser(subparsers):
    parser = subparsers.add_parser('segment-char-ngrams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="segment rare words into character n-grams")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--vocab', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="Vocabulary file.")
    parser.add_argument(
        '--shortlist', type=int, metavar='INT', default=0,
        help="do not segment INT most frequent words in vocabulary (default: '%(default)s')).")
    parser.add_argument(
        '-n', type=int, metavar='INT', default=2,
        help="segment rare words into character n-grams of size INT (default: '%(default)s')).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")

    return parser


def create_learn_joint_bpe_and_vocab_parser(subparsers):
    parser = subparsers.add_parser('learn-joint-bpe-and-vocab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), required=True, nargs = '+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), required=True,
        metavar='PATH',
        help="Output file for BPE codes.")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--write-vocabulary', type=argparse.FileType('w'), nargs = '+', default=None,
        metavar='PATH', dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="subword-nmt segmentation")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    learn_bpe_parser = create_learn_bpe_parser(subparsers)
    apply_bpe_parser = create_apply_bpe_parser(subparsers)
    get_vocab_parser = create_get_vocab_parser(subparsers)
    segment_char_ngrams_parser = create_segment_char_ngrams_parser(subparsers)
    learn_joint_bpe_and_vocab_parser = create_learn_joint_bpe_and_vocab_parser(subparsers)

    args = parser.parse_args()

    if args.command == 'learn-bpe':
        # read/write files as UTF-8
        if args.input.name != '<stdin>':
            args.input = codecs.open(args.input.name, encoding='utf-8')
        if args.output.name != '<stdout>':
            args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

        learn_bpe(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)
    elif args.command == 'apply-bpe':
        # read/write files as UTF-8
        args.codes = codecs.open(args.codes.name, encoding='utf-8')
        if args.input.name != '<stdin>':
            args.input = codecs.open(args.input.name, encoding='utf-8')
        if args.output.name != '<stdout>':
            args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
        if args.vocabulary:
            args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

        if args.vocabulary:
            vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
        else:
            vocabulary = None

        bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries)

        for line in args.input:
            args.output.write(bpe.segment(line).strip())
            args.output.write('\n')
    elif args.command == 'get-vocab':
        get_vocab(args.train_file, args.vocab_file)
    elif args.command == 'segment-char-ngrams':
        segment_char_ngrams(args)
    elif args.command == 'learn-joint-bpe-and-vocab':
        learn_joint_bpe_and_vocab(args)
    else:
        raise Exception('Invalid command provided')


if __name__ == '__main__':
    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    main()
