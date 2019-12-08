# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Baseline(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(Baseline, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        # self.word_embeddings = ModelEmbeddings(embed_file, embedding_length, vocab_size)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.classifier = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
        self.W_s1 = nn.Linear(2*hidden_size, self.output_size)
        self.relu = nn.ReLU()
        # self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input, batch_size=None):
        # print("shape 1")
        # print(input.shape)
        input1 = self.word_embeddings(input)
        # print("shape 2")
        # print(input1.shape)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, 1, self.hidden_size))
            c_0 = Variable(torch.zeros(2, 1, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(2, 1, self.hidden_size))
            c_0 = Variable(torch.zeros(2, 1, self.hidden_size))
        output1, _ = self.classifier(input1, (h_0, c_0))
        # print("shape 3")
        # print(output1.shape)
        output1 = self.W_s1(output1)
        # print("shape 4")
        # print(output1.shape)
        output2 = self.relu(output1)
        # print("shape 5")
        # print(output2.shape)
        return output2
