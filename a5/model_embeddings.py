#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.e_char = 50
        self.w_word = 21  # same as max_word_length. same value (21) used in function pad_sents_char in utils.py
        self.embed_size = embed_size  # same as e_word


        self.char_embedding = nn.Embedding(len(vocab.char2id), self.e_char, vocab.char2id['<pad>'])
        self.cnn = CNN(self.e_char, self.embed_size, self.w_word)
        self.highway = Highway(self.embed_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        # x_padded has shape : (sentence_length, batch_size, max_word_length)
        x_padded = input

        # x_emb has shape : (sentence_length, batch_size, max_word_length, e_char)
        x_emb = self.char_embedding(x_padded)

        # x_reshape_4D has shape : (sentence_length, batch_size, e_char, max_word_length)
        x_reshape_4D = x_emb.permute(0, 1, 3, 2)

        sentence_length, batch_size, e_char, max_word_length = x_reshape_4D.shape

        # x_reshape has shape : (-1, e_char, max_word_length)
        x_reshape = x_reshape_4D.view(-1, e_char, max_word_length)

        # x_conv_out has shape : (-1, e_word)
        x_conv_out = self.cnn(x_reshape)

        # x_word_embed has shape : (-1, e_word)
        x_word_embed = self.highway(x_conv_out)

        output = x_word_embed.view(sentence_length, batch_size, self.embed_size)

        return output

        ### END YOUR CODE

