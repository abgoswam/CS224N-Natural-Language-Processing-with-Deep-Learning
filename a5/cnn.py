#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char, f, m_word, k=5):
        """
        :param e_char: (int): Embedding size for character (dimensionality)
        :param f: (int): number of filters, also called [number of output features] or [number of output channels]
                         in this application f == e_word (aka size of word embedding embed_size)
        :param m_word (int)
        :param k: (int): kernel size, also called [window size] which dictates the size of window used to compute features

        """
        super(CNN, self).__init__()

        # self.conv1 (Conv1D layer with bias), called W in the PDF.
        # self.maxpool (MaxPool) layer

        self.conv1 = torch.nn.Conv1d(in_channels=e_char, out_channels=f, kernel_size=k)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=(m_word-k+1))


    def forward(self, x_reshaped):
        """

        :param x_reshaped: tensor of shape (batch_size, e_char, m_word)
        :return: x_conv_out: tensor of shape (batch_size, e_word)
        """

        # x_conv shape : (batch_size, f, (m_word-k+1))
        x_conv = self.conv1(x_reshaped)

        # x_conv_out_3d shape : (batch_size, f, 1)
        x_conv_out_3d = self.maxpool(F.relu(x_conv))

        # Note:
        # When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.

        # Use the following docs to implement this functionality:
        #     Tensor Dimension Squeezing:
        #         https://pytorch.org/docs/stable/torch.html#torch.squeeze


        # x_conv_out shape : (batch_size, f)
        #                    in this application f == e_word. So shape of  x_conv_out should be (batch_size, e_word)
        x_conv_out = torch.squeeze(x_conv_out_3d, 2)

        return x_conv_out

### END YOUR CODE

