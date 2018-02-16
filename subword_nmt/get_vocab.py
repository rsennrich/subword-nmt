from __future__ import print_function

import os
from collections import Counter

def get_vocab(train_file, vocab_file):
    c = Counter()

    with open(train_file, 'r', encoding='utf-8') as file_in:
        lines = file_in.read().splitlines()

    for line in lines:
        for word in line.split():
            c[word] += 1

    with open(vocab_file, 'w', encoding='utf-8') as file_out:
        for key, f in sorted(c.items(), key=lambda x: x[1], reverse=True):
            file_out.write(key + " " + str(f) + os.linesep)
