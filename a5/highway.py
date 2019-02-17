#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, e_word, dropout_rate=0.3):
        """
        :param e_word: (int): Embedding size (dimensionality)
        :param dropout_rate: (float): Dropout probability, for highway

        """
        super(Highway, self).__init__()
        self.dropout_rate = dropout_rate

        # self.projection (Linear Layer with bias), called W_{proj} in the PDF.
        # self.gate (Linear Layer with bias), called W_{gate} in the PDF.
        # self.sigmoid (Sigmoid) being applied to W_{gate}
        # self.dropout (Dropout Layer)

        self.projection = torch.nn.Linear(in_features=e_word,
                                          out_features=e_word,
                                          bias=True)

        self.gate = torch.nn.Linear(in_features=e_word,
                                    out_features=e_word,
                                    bias=True)

        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)


    def forward(self, x_conv_out):
        """

        :param x_conv_out: tensor of shape (batch_size, e_word)
        :return: x_word_embed . tensor of shape (batch_size, e_word)
        """

        # x_proj, x_gate, x_highway shape : (batch_size, e_word)

        x_proj = F.relu(self.projection(x_conv_out))
        x_gate = self.sigmoid(self.gate(x_conv_out))
        x_highway = (x_gate * x_proj) + ((1 - x_gate) * x_conv_out)
        x_word_embed = self.dropout(x_highway)
        return x_word_embed

### END YOUR CODE
