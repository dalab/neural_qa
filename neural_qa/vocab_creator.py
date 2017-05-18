#! /usr/bin/env python
# coding=utf-8

from nltk.tokenize.toktok import ToktokTokenizer
from tensorflow.contrib import learn
import numpy as np
import data_helpers

max_len = 0


def my_tokenizer(iterator):
    global max_len
    tknzr = ToktokTokenizer()
    for value in iterator:
        value = value.replace('-', " - ")
        value = value.replace('/', " / ")
        value = value.lower()
        value = tknzr.tokenize(value)
        max_len = max(max_len, len(value))
        yield value

# Load (question, tree, label)
x_q_raw, x_t_raw = data_helpers.load_q_t("./data/example.tsv")

# Build Vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(127, tokenizer_fn=my_tokenizer)
vocab_processor.fit(np.r_[x_q_raw, x_t_raw])
print(max_len)
vocab_processor.save("vocab_new_tokens")
