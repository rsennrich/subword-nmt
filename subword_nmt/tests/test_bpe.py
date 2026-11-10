#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
import codecs

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from learn_bpe import learn_bpe
from apply_bpe import BPE


class TestBPELearnMethod(unittest.TestCase):

    def test_learn_bpe(self):
        infile = codecs.open(os.path.join(currentdir,'data','corpus.en'), encoding='utf-8')
        outfile = codecs.open(os.path.join(currentdir,'data','bpe.out'), 'w', encoding='utf-8')
        learn_bpe(infile, outfile, 1000)
        infile.close()
        outfile.close()

        outlines = open(os.path.join(currentdir,'data','bpe.out'))
        reflines = open(os.path.join(currentdir,'data','bpe.ref'))

        for line, line2 in zip(outlines, reflines):
            self.assertEqual(line, line2)

        outlines.close()
        reflines.close()

class TestBPESegmentMethod(unittest.TestCase):

    def setUp(self):

        with codecs.open(os.path.join(currentdir,'data','bpe.ref'), encoding='utf-8') as bpefile:
            self.bpe = BPE(bpefile)

        self.infile = codecs.open(os.path.join(currentdir,'data','corpus.en'), encoding='utf-8')
        self.reffile = codecs.open(os.path.join(currentdir,'data','corpus.bpe.ref.en'), encoding='utf-8')

    def tearDown(self):

        self.infile.close()
        self.reffile.close()

    def test_apply_bpe(self):

        for line, ref in zip(self.infile, self.reffile):
            out = self.bpe.process_line(line)
            self.assertEqual(out, ref)

    def test_trailing_whitespace(self):
        """BPE.proces_line() preserves leading and trailing whitespace"""

        orig = '  iron cement  \n'
        exp = '  ir@@ on c@@ ement  \n'

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)

    def test_utf8_whitespace(self):
        """UTF-8 whitespace is treated as normal character, not word boundary"""

        orig = 'iron\xa0cement\n'
        exp = 'ir@@ on@@ \xa0@@ c@@ ement\n'

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)

    def test_empty_line(self):

        orig = '\n'
        exp = '\n'

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)

    def test_no_pairs_does_not_crash(self):
        """learn_bpe must not crash when there are no learnable pairs (empty pair stats)."""
        import io
        infile = io.StringIO('a\nb\nc\nd\n')
        outfile = io.StringIO()
        learn_bpe(infile, outfile, num_symbols=10)
        self.assertEqual(outfile.getvalue(), '#version: 0.2\n')

    def test_min_frequency_above_all_pairs(self):
        """learn_bpe must stop gracefully when min_frequency exceeds every pair frequency."""
        import io
        infile = io.StringIO('a b\nc d\n')
        outfile = io.StringIO()
        learn_bpe(infile, outfile, num_symbols=5, min_frequency=2)
        self.assertEqual(outfile.getvalue(), '#version: 0.2\n')
if __name__ == '__main__':
    unittest.main()
