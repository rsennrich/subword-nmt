#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import codecs

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import learn_bpe
from apply_bpe import BPE


class TestBPELearnMethod(unittest.TestCase):

    def test_learn_bpe(self):
        infile = codecs.open(os.path.join(currentdir,'data','corpus.en'), encoding='utf-8')
        outfile = codecs.open(os.path.join(currentdir,'data','bpe.out'), 'w', encoding='utf-8')
        learn_bpe.main(infile, outfile, 1000)
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

if __name__ == '__main__':
    unittest.main()