'''
Author: Sudharsansai, UCLA
Evaluation Helper functions for n-gram LM
'''



import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle


def eval_loss(model, inputs, outputs, criterion):
	model.eval()
	hidden = model.init_hidden(inputs.shape[0])
	pred, _ = model(inputs, hidden)
	model.zero_grad()
	return criterion(pred, outputs).data[0]

def decode(model, start, idx_to_word, max_tokens):
	model.eval()
	print(idx_to_word[start], end=' ')
	start = np.asarray([[start]])
	hidden = model.init_hidden(1)
	for i in range(max_tokens):
		pred, hidden = model(Variable(torch.LongTensor(start)).cuda(), hidden)
		pred = torch.squeeze(pred, dim=0)
		#print('PRED SHAPE: ', pred.shape, 'DICT SIZE: ', len(idx_to_word))
		assert pred.shape[0] == len(idx_to_word)
		start = pred.max(dim=0)[1]
		print(idx_to_word[start.data[0]], end=' ')
		start = np.asarray([[start.data[0]]])
	model.zero_grad()
	print("\n")

def perplexity(model, inputs, outputs, criterion):
	model.eval()
	hidden = model.init_hidden(inputs.shape[0])
	pred, _ = model(inputs, hidden)
	model.zero_grad()
	return np.power(2, criterion(pred, outputs).data[0])