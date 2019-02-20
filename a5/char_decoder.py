#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()

        self.charDecoder = torch.nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)

        self.char_output_projection = torch.nn.Linear(in_features=hidden_size,
                                                      out_features=len(target_vocab.char2id),
                                                      bias=True)

        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, target_vocab.char2id['<pad>'])

        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        # This should be of shape (length, batch, char_embedding_size)
        X = self.decoderCharEmb(input)

        # output is of shape (length, batch, hidden_size)

        # dec_hidden is tuple of two tensors (h_n, c_n)
        # h_n (1, batch, hidden_size): tensor containing the hidden state for t = seq_len
        # c_n (1, batch, hidden_size): tensor containing the cell state for t = seq_len

        output, dec_hidden = self.charDecoder(X, dec_hidden)

        scores = self.char_output_projection(output)

        return scores, dec_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # tensor of shape (length - 1, batch)
        # we want it this to be our target
        target = char_sequence[1:]

        # target would be of shape (length - 1)*batch
        target_CE = target.contiguous().view(-1)


        # tensor of shape (length - 1, batch)
        # this is our source we want to feed to the RNN
        source = char_sequence[:-1]

        # scores would be of shape (length - 1, batch, self.vocab_size)
        # output scores of RNN which reflects what the RNN thinks is output
        scores, dec_hidden = self.forward(source, dec_hidden)

        # scores_CE would be of shape (N, C)
        # where
        # N = (length - 1)*batch
        # C = self.vocab_size
        scores_CE = scores.view(-1, len(self.target_vocab.char2id))

        # padding characters should not contribute to the cross-entropy loss.
        # we want to compute sum (not average) across entire batch

        loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char2id['<pad>'])
        loss_CE = loss(scores_CE, target_CE)

        return loss_CE

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        # dec_hidden would be a tuple of two tensors of size (1, batch, hidden_size)
        dec_hidden = initialStates

        # initialize current_char_input shape (length, batch) . Here length == 1
        _, batch_size, hidden_size = initialStates[0].shape
        current_char = [self.target_vocab.start_of_word for _ in range(batch_size)]
        current_char_tensor = torch.tensor(current_char, device=device)

        # current_char_input shape(length, batch). Here length == 1
        current_char_input = current_char_tensor.view(1, -1)

        # output_word_stack is of shape (max_length, batch).
        output_word_stack = None

        for t in range(max_length):
            # scores is of shape (length, batch, self.vocab_size). Here length == 1
            scores, dec_hidden = self.forward(current_char_input, dec_hidden)

            softmax = torch.nn.Softmax(dim=-1)
            probabilities = softmax(scores)  # so probabilities has shape (1, batch, self.vocab_size)
            current_char_input = probabilities.argmax(dim=-1)  # shape(length, batch). Here length == 1

            if output_word_stack is None:
                output_word_stack = current_char_input
            else:
                output_word_stack = torch.cat((output_word_stack, current_char_input))

        # transpose so we get output_words (batch, max_length).
        output_word_tensors = output_word_stack.t()

        decodedWords = []
        for word_tensor in output_word_tensors:
            word = ''
            for c_tensor in word_tensor:
                c_id = c_tensor.item()
                if c_id == self.target_vocab.end_of_word:
                    break
                word += self.target_vocab.id2char[c_id]
            decodedWords.append(word)

        return decodedWords

        ### END YOUR CODE

