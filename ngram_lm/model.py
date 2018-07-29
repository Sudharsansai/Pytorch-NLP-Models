'''
Author: Sudharsansai, UCLA
Model definition for n-gram LM
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle

dtype = torch.cuda.FloatTensor

class LSTM_LM(nn.Module):  # inheriting from nn.Module!

	def __init__(self, vocab_size, emb_dim, hidden_dim, batch_size):
		super(LSTM_LM, self).__init__()
		self.hidden_dim = hidden_dim
		self.emb_dim = emb_dim
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.embedding = nn.Embedding(vocab_size, emb_dim)
		self.lstm = nn.LSTM(emb_dim, hidden_dim)
		self.linear = nn.Linear(hidden_dim, vocab_size)

	def init_hidden(self, batch_size):
		# for h, c
		return (Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = True).type(dtype),
				Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = True).type(dtype))

	def forward(self, inputs, hidden):
		# inputs is of shape (BS, CONTEXT_LEN)
		#print('INPUTS SHAPE: ', inputs.shape)
		batch_size = inputs.shape[0]

		inputs = inputs.transpose(0,1)
		# inputs is of shape (CONTEXT_LEN, BS)

		embedded = self.embedding(inputs)
		# embedded is of shape (CONTEXT_LEN, BS, emb_dim)

		outputs, hidden = self.lstm(embedded, hidden)
		# shape of outputs: (context_len, bs, hidden_dim*num_dirs(=1))
		# shape of hidden: ((num_layers*num_dirs(=1), bs, hidden_dim), (num_layers*num_dirs(=1), bs, hidden_dim))

		output = torch.squeeze(hidden[0], 0)
		# output has shape (bs, hidden_dim)

		output = self.linear(output)
		# output has shape (bs, vocab_size)

		assert output.shape==(embedded.shape[1], self.vocab_size)

		return F.log_softmax(output, dim=1), hidden

