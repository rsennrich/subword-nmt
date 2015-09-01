#! /usr/bin/env python

import sys
from collections import Counter

c = Counter()

for line in sys.stdin:
    for word in line.split():
        c[word] += 1

for key,f in sorted(c.items(), key=lambda x: x[1], reverse=True):
    print key, f
