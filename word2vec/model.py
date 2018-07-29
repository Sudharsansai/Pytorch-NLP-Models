'''
Author: Sudharsansai, UCLA
Model definition for word2vec-CBOW
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize

class CBOW(nn.Module):  # inheriting from nn.Module!

	def __init__(self, vocab_size, emb_size):
		super(CBOW, self).__init__()
		self.embedding = nn.Embedding(vocab_size, emb_size)
		self.linear = nn.Linear(emb_size, vocab_size)


	def forward(self, inputs):
		# inputs is of shape (BS, CONTEXT_SIZE)
		embedded = self.embedding(inputs)
		# embedded is of shape (BS, CONTEXT_SIZE, emb_size)
		embedded = torch.sum(embedded, dim=1)
		# embedded is of shape (BS, emb_size)
		output = self.linear(embedded)
		# output is of shape (BS, vocab_size)
		softmax_output = F.log_softmax(output, dim=1)
		# softmax_output is of shape (BS, vocab_size)
		return softmax_output	