# Author: Rico Sennrich

from __future__ import unicode_literals, division


def segment_char_ngrams(args):
    # read/write files as UTF-8
    args.vocab = codecs.open(args.vocab.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    vocab = [line.split()[0] for line in args.vocab if len(line.split()) == 2]
    vocab = dict((y,x) for (x,y) in enumerate(vocab))

    for line in args.input:
        for word in line.split():
            if word not in vocab or vocab[word] > args.shortlist:
                i = 0
                while i*args.n < len(word):
                    args.output.write(word[i*args.n:i*args.n+args.n])
                    i += 1
                    if i*args.n < len(word):
                        args.output.write(args.separator)
                    args.output.write(' ')
            else:
                args.output.write(word + ' ')

        args.output.write('\n')
