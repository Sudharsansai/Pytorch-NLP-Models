'''
Author: Sudharsansai, UCLA
Dataloader Helper functions for word2vec
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize


raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""
raw_text = word_tokenize(raw_text)

def create_dataset(CONTEXT_SIZE):
	data = [([raw_text[i-j] for j in range(CONTEXT_SIZE)]+[raw_text[i+j] for j in range(CONTEXT_SIZE)], 
		raw_text[i]) for i in range(CONTEXT_SIZE, len(raw_text)-CONTEXT_SIZE)]
	shuffle(data)
	return data

def split_dataset(data, split_ratio):
	train_split = int(split_ratio*len(data))
	train_data = data[:train_split]
	test_data = data[train_split:]
	return train_data, test_data

def create_dictionary():
	vocab = set(raw_text)
	word_to_ix = {word:i+1 for i, word in enumerate(vocab)}
	word_to_ix['UNK'] = 0
	del vocab
	return word_to_ix


def make_batch(data, batch_size):

	shuffle(data)

	if(batch_size==-1):
		# if batch size is -1, return the complete dataset as a batch
		return [example[0] for example in data], [[example[1]] for example in data]

	ret_data, ret_labels = [], []
	for i in range(0, len(data)-batch_size, batch_size):
		ret_data.append([inp[0] for inp in data[i:i+batch_size]])
		ret_labels.append([[inp[1]] for inp in data[i:i+batch_size]])
	return ret_data, ret_labels


def vectorize_data(data, word_to_ix):
	# data is of shape (batch_size, num_words_per_example)
	bow_vector=[[word_to_ix.get(word, 0) for word in example] for example in data]
	return torch.LongTensor(np.asarray(bow_vector))