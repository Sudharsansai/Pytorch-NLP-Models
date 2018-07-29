'''
Author: Sudharsansai, UCLA
Helper functions for Char Level POS Tagger, which optionally takes a word level component
'''

import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize

train_file = "train.txt"
test_file = "val.txt"

dtype = torch.cuda.LongTensor


def process_data(directory, file):
	data = []
	with open(os.path.join("./data", directory, file), "r", encoding="utf-8") as inp:
		for line in inp:
			line = line.split()
			if(len(line[:-1])!=0):
				data.append(line[:-1])  	# each line in the files is terminated by a period
		inp.close()
	#shuffle(data)
	return data

def create_dataset(directory):
	train_data = process_data(directory, train_file)
	print("Training dataset created...")
	test_data = process_data(directory, test_file)
	print("Testing dataset created...")
	return train_data, test_data

def create_dictionary(data, threshold=0):
	# data is a list of lists
	if(threshold==0):
		# if there is no threshold
		vocab = set()
		for sentence in data:
			for word in sentence:
				vocab.add(word)
		vocab = {word:(i+1) for i, word in enumerate(vocab)}
		vocab["UNK"] = 0
		return vocab

	vocab = {}
	for line in data:
		for word in line:
			if(word in vocab):
				vocab[word]+=1
			else:
				vocab[word]=1
	dictionary = {}
	dictionary["UNK"] = 0

	for word, count in vocab.items():
		if(count>=threshold) and (word not in dictionary):
			dictionary[word] = len(dictionary)
	del vocab

	return dictionary


def vectorize_inputs(data, word_dict, char_dict):
	vectorized_data = []
	for sentence in data:
		vectorized_sent = [word_dict.get(word, 0) for word in sentence]
		vectorized_chars = [[char_dict.get(char,0) for char in word] for word in sentence]
		vectorized_data.append((vectorized_chars, vectorized_sent))

	for i in range(len(vectorized_data)):
		example = vectorized_data[i]
		for j in range(len(example[0])):
			chars = example[0][j]
			if(len(chars)==0):
				print(data[i][j])
				assert False
	print("Printing Vectorized Data: ")
	print("Word id list: ")
	for word in vectorized_data[1]:
		print(word)
	print("Each following line corresponds to a list of char ids")
	for chars in vectorized_data[0]:
		print(chars)

	return vectorized_data

def vectorize_tags(data, tags_dict):
	return [[tags_dict.get(tag,0) for tag in example] for example in data]