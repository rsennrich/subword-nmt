Subword Neural Machine Translation
==================================

This repository contains preprocessing scripts to segment text into subword
units. The primary purpose is to facilitate the reproduction of our experiments
on Neural Machine Translation with subword units (see below for reference).

USAGE INSTRUCTIONS
------------------

Check the individual files for usage instructions.

To apply byte pair encoding to word segmentation, invoke these commands:

    ./learn_bpe.py -s {num_operations} < {train_file} > {codes_file}
    ./apply_bpe.py -c {codes_file} < {test_file}

To segment rare words into character n-grams, do the following:

    ./get_vocab.py < {train_file} > {vocab_file}
    ./segment-char-ngrams.py --vocab {vocab_file} -n {order} --shortlist {size} < {test_file}

The original segmentation can be restored with a simple replacement:

    sed "s/@@ //g"

PUBLICATIONS
------------

The segmentation methods are described in:

Rico Sennrich, Barry Haddow and Alexandra Birch (2016):
    Neural Machine Translation of Rare Words with Subword Units
    Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.