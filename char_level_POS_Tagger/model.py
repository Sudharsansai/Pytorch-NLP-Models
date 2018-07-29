'''
Author: Sudharsansai, UCLA
Model definition for Char Level POS Tagger which optionally takes a word level component

Important: This optionality can be manipulated by the last 3 arguments on the constructor for the model.
They should all be either -1 or none-(-1)
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle

dtype = torch.cuda.FloatTensor

class POSTagger(nn.Module):  # inheriting from nn.Module!

	def __init__(self, target_size, char_vocab_size, char_emb_dim, char_hidden_dim, word_vocab_size=-1, word_emb_dim=-1, word_hidden_dim=-1):
		super(POSTagger, self).__init__()
		self.word_hidden_dim = word_hidden_dim
		self.word_emb_dim = word_emb_dim
		self.word_vocab_size = word_vocab_size

		self.char_hidden_dim = char_hidden_dim
		self.char_emb_dim = char_emb_dim
		self.char_vocab_size = char_vocab_size

		self.target_size = target_size

		self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
		self.char_lstm = nn.LSTM(char_emb_dim, char_hidden_dim)

		if(word_vocab_size!=-1):
			self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
			self.word_lstm = nn.LSTM(word_emb_dim, word_hidden_dim)
			self.linear = nn.Linear(char_hidden_dim+word_hidden_dim, target_size)
		else:
			self.linear = nn.Linear(char_hidden_dim, target_size)

		self.activation = nn.ReLU()

	def init_hidden(self, dim):
		# for h, c
		return (Variable(torch.zeros(1, 1, dim), requires_grad = True).type(dtype),
				Variable(torch.zeros(1, 1, dim), requires_grad = True).type(dtype))

	def forward(self, sentence, char_ids):
		# sentence is a list of size (seq_len) each being a word_id
		# char_ids is a list of (list: char ids for a word)

		char_hiddens = []

		if(len(char_ids)==0):
			print("Number of words in the sentence: ", len(char_ids))
		for word in char_ids:
			char_hidden = self.init_hidden(self.char_hidden_dim)
			chars = Variable(torch.cuda.LongTensor(np.asarray(word)).view(-1, 1), requires_grad=False)
			chars = self.char_embedding(chars)
			chars = self.activation(chars)
			# char is of shape (word_len, 1, char_embedding_dim)
			_, char_hidden = self.char_lstm(chars, char_hidden)
			# char_hidden[0] is the final hidden state of shape (1, 1, char_hidden_dim)
			char_hiddens.append(char_hidden[0])

		char_hiddens = torch.cat(char_hiddens, dim=0)
		# char_hiddens is of shape (seq_len, 1, char_hidden_dim)

		if(self.word_vocab_size!=-1):
			word = self.word_embedding(Variable(torch.cuda.LongTensor(np.asarray(sentence)).view(-1,1), requires_grad=False))
			# word is of shape (seq_len, 1, word_embedding_dim)
			word = self.activation(word)
			word_hidden = self.init_hidden(self.word_hidden_dim)
			word_out, _ = self.word_lstm(word, word_hidden)
			# word_out is of shape (seq_len, 1, word_hidden_dim)
			hidden = torch.cat((word_out, char_hiddens), dim=2)
		else:
			hidden = char_hiddens

		# hidden is of shape (seq_len, 1, word_hidden_dim+char_hidden_dim) or of shape (seq_len, 1, char_hidden_dim)
		if(self.word_vocab_size!=-1):
			assert hidden.shape==(len(sentence), 1, self.word_hidden_dim+self.char_hidden_dim)
		else:
			assert hidden.shape==(len(sentence), 1, self.char_hidden_dim)

		# output mapping
		outputs = self.linear(hidden)
		# outputs is of shape (seq_len, 1, target_size)

		return F.log_softmax(outputs, dim=2).transpose(0,1).transpose(1,2), hidden

