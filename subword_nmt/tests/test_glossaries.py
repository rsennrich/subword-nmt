#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import mock

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from apply_bpe import isolate_glossary, BPE

class TestIsolateGlossaryFunction(unittest.TestCase):

    def setUp(self):
        self.glossary = 'like'

    def _run_test_case(self, test_case):
        orig, expected = test_case
        out = isolate_glossary(orig, self.glossary)
        self.assertEqual(out, expected)

    def test_empty_string(self):
        orig = ''
        exp = ['']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_no_glossary(self):
        orig = 'word'
        exp = ['word']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_isolated_glossary(self):
        orig = 'like'
        exp = ['like']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_word_one_side(self):
        orig = 'likeword'
        exp = ['like', 'word']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_words_both_sides(self):
        orig = 'wordlikeword'
        exp = ['word', 'like', 'word']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_back_to_back_glossary(self):
        orig = 'likelike'
        exp = ['like', 'like']
        test_case = (orig, exp)
        self._run_test_case(test_case)

    def test_multiple_glossaries(self):
        orig = 'wordlikewordlike'
        exp = ['word', 'like', 'word', 'like']
        test_case = (orig, exp)
        self._run_test_case(test_case)

class TestBPEIsolateGlossariesMethod(unittest.TestCase):

    def setUp(self):

        amock = mock.MagicMock()
        amock.readline.return_value = 'something'
        glossaries = ['like', 'Manuel', 'USA']
        self.bpe = BPE(amock, glossaries=glossaries)

    def _run_test_case(self, test_case):
        orig, expected = test_case
        out = self.bpe._isolate_glossaries(orig)
        self.assertEqual(out, expected)

    def test_multiple_glossaries(self):
        orig = 'wordlikeUSAwordManuelManuelwordUSA'
        exp = ['word', 'like', 'USA', 'word', 'Manuel', 'Manuel', 'word', 'USA']
        test_case = (orig, exp)
        self._run_test_case(test_case)

class TestRegexIsolateGlossaries(unittest.TestCase):

    def setUp(self):

        amock = mock.MagicMock()
        amock.readline.return_value = 'something'
        glossaries = ["<country>\w*</country>", "<name>\w*</name>", "\d+"]
        self.bpe = BPE(amock, glossaries=glossaries)

    def _run_test_case(self, test_case):
        orig, expected = test_case
        out = self.bpe._isolate_glossaries(orig)
        self.assertEqual(out, expected)

    def test_regex_glossaries(self):
        orig = 'wordlike<country>USA</country>word10001word<name>Manuel</name>word<country>USA</country>'
        exp = ['wordlike', '<country>USA</country>', 'word', '10001', 'word', '<name>Manuel</name>', 'word', '<country>USA</country>']
        test_case = (orig, exp)
        self._run_test_case(test_case) 

def encode_mock(segment, x2, x3, x4, x5, x6, x7, glosses, dropout):
    if glosses.match(segment):
        return (segment,)
    else:
        l = len(segment)
        return (segment[:l//2], segment[l//2:])

class TestBPESegmentMethod(unittest.TestCase):

    def setUp(self):

        amock = mock.MagicMock()
        amock.readline.return_value = 'something'
        glossaries = ['like', 'Manuel', 'USA']
        self.bpe = BPE(amock, glossaries=glossaries)

    @mock.patch('apply_bpe.encode', side_effect=encode_mock)
    def _run_test_case(self, test_case, encode_function):

        orig, expected = test_case
        out = self.bpe.segment(orig)

        self.assertEqual(out, expected)

    def test_multiple_glossaries(self):
        orig = 'wordlikeword likeManuelword'
        exp = 'wo@@ rd@@ like@@ wo@@ rd like@@ Manuel@@ wo@@ rd'
        test_case = (orig, exp)
        self._run_test_case(test_case)

if __name__ == '__main__':
    unittest.main()
