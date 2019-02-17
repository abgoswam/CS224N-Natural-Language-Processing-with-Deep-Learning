#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21
    # max_word_length = 5

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents() 
    ###     method below using the padding character from the arguments. You should ensure all 
    ###     sentences have the same number of words and each word has the same number of 
    ###     characters. 
    ###     Set padding words to a `max_word_length` sized vector of padding characters.  
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles 
    ###     padding and unknown words.

    max_sentence_length = 0
    for s in sents:
        if len(s) > max_sentence_length:
            max_sentence_length = len(s)

    sents_padded = []

    for s in sents:
        s_padded = []
        for w in s:
            length_word = len(w)
            if length_word > max_word_length:
                terminating_val = w[-1]
                wnew = w[:max_word_length]
                wnew[-1] = terminating_val
            else:
                wpad = [char_pad_token for _ in range(max_word_length - length_word)]
                wnew = w + wpad
            s_padded.append(wnew)

        sents_padded.append(s_padded)

    for s in sents_padded:
        dummy_words = [[char_pad_token for _ in range(max_word_length)] for _ in range(max_sentence_length - len(s))]
        s.extend(dummy_words)

    ### END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def check_pad_sents_char():
    sents = [
        [[1, 3, 2]],
        [[1, 3, 3, 3, 3, 2], [1, 3, 3, 3, 2], [1, 3, 3, 2]],
        [[1, 3, 3, 3, 3, 2], [1, 3, 3, 3, 2]],
        [[1, 3, 2], [1, 2]],
    ]

    for s in sents:
        print(s)
    print("-----")

    sents_padded = pad_sents_char(sents, 9)
    for s in sents:
        print(s)
    print("-----")

    for s in sents_padded:
        print(s)
    print("-----")


if __name__ == '__main__':
    check_pad_sents_char()