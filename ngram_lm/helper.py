'''
Author: Sudharsansai, UCLA
Helper functions for n-gram Language Model
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize

train_file = "./data/ptb.train.txt"
test_file = "./data/ptb.test.txt"
valid_file = "./data/ptb.valid.txt"

dtype = torch.cuda.LongTensor

def process_data(context_size, file):
	data = []
	with open(file, "r", encoding="utf-8") as inp:
		for line in inp:
			line = line.split()
			data.extend([([line[i+j] for j in range(context_size)], line[i+context_size]) for i in range(len(line)-context_size)])
		inp.close()
	shuffle(data)
	return data

def create_dataset(context_size):
	train_data = process_data(context_size, train_file)
	test_data = process_data(context_size, test_file)
	valid_data = process_data(context_size, valid_file)
	return train_data, test_data, valid_data

def create_dictionary(data):
	vocab = set()
	for (inp, lab) in data:
		for word in inp:
			vocab.add(word)
		vocab.add(lab)
	word_to_ix = {word:i+1 for i, word in enumerate(vocab)}
	idx_to_word = {(i+1):word for i, word in enumerate(vocab)}
	word_to_ix['UNK'] = 0
	idx_to_word[0] = "UNK"
	del vocab
	return word_to_ix, idx_to_word


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
	return torch.Tensor(np.asarray(bow_vector)).type(dtype)