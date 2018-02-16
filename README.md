Subword Neural Machine Translation
==================================

This repository contains preprocessing scripts to segment text into subword
units. The primary purpose is to facilitate the reproduction of our experiments
on Neural Machine Translation with subword units (see below for reference).

INSTALLATION
------------------
    pip install https://github.com/rsennrich/subword-nmt/archive/master.zip

USAGE INSTRUCTIONS
------------------

Check the individual files for usage instructions.

To apply byte pair encoding to word segmentation, invoke these commands:

    subword-nmt learn-bpe -s {num_operations} < {train_file} > {codes_file}
    subword-nmt apply-bpe -c {codes_file} < {test_file} > {out_file}

To segment rare words into character n-grams, do the following:

    subword-nmt get-vocab --train_file {train_file} --vocab_file {vocab_file}
    subword-nmt segment-char-ngrams --vocab {vocab_file} -n {order} --shortlist {size} < {test_file} > {out_file}

The original segmentation can be restored with a simple replacement:

    sed -r 's/(@@ )|(@@ ?$)//g'


BEST PRACTICE ADVICE FOR BYTE PAIR ENCODING IN NMT
--------------------------------------------------

We found that for languages that share an alphabet, learning BPE on the
concatenation of the (two or more) involved languages increases the consistency
of segmentation, and reduces the problem of inserting/deleting characters when
copying/transliterating names.

However, this introduces undesirable edge cases in that a word may be segmented
in a way that has only been observed in the other language, and is thus unknown
at test time. To prevent this, `apply_bpe.py` accepts a `--vocabulary` and a
`--vocabulary-threshold` option so that the script will only produce symbols
which also appear in the vocabulary (with at least some frequency).

To use this functionality, we recommend the following recipe (assuming L1 and L2
are the two languages):

Learn byte pair encoding on the concatenation of the training text, and get resulting vocabulary for each:

    cat {train_file}.L1 {train_file}.L2 | subword-nmt learn-bpe -s {num_operations} -o {codes_file}
    subword-nmt apply-bpe -c {codes_file} < {train_file}.L1 && subword-nmt get-vocab --train_file {codes_file} --vocab_file {vocab_file}.L1
    subword-nmt apply-bpe -c {codes_file} < {train_file}.L2 && subword-nmt get-vocab --train_file {codes_file} --vocab_file {vocab_file}.L2

more conventiently, you can do the same with with this command:

    subword-nmt learn-joint-bpe-and-vocab --input {train_file}.L1 {train_file}.L2 -s {num_operations} -o {codes_file} --write-vocabulary {vocab_file}.L1 {vocab_file}.L2

re-apply byte pair encoding with vocabulary filter:

    subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.L1 --vocabulary-threshold 50 < {train_file}.L1 > {train_file}.BPE.L1
    subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.L2 --vocabulary-threshold 50 < {train_file}.L2 > {train_file}.BPE.L2

as a last step, extract the vocabulary to be used by the neural network. Example with Nematus:

    nematus/data/build_dictionary.py {train_file}.BPE.L1 {train_file}.BPE.L2

[you may want to take the union of all vocabularies to support multilingual systems]

for test/dev data, re-use the same options for consistency:

    subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.L1 --vocabulary-threshold 50 < {test_file}.L1 > {test_file}.BPE.L1


PUBLICATIONS
------------

The segmentation methods are described in:

Rico Sennrich, Barry Haddow and Alexandra Birch (2016):
    Neural Machine Translation of Rare Words with Subword Units
    Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

ACKNOWLEDGMENTS
---------------
This project has received funding from Samsung Electronics Polska sp. z o.o. - Samsung R&D Institute Poland, and from the European Union’s Horizon 2020 research and innovation programme under grant agreement 645452 (QT21).
