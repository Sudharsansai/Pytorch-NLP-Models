'''
Author: Sudharsansai, UCLA
Evaluation Helper functions for Char Level POS Tagger, which optionally takes a word level component
'''



import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
import helper
from torch.distributions.categorical import Categorical


def eval_loss(model, inputs, outputs, criterion):
	# inputs is a list of (example: list of (word: tuples of the form (word_id, list of character_ids)))
	# outputs is a list of (example: list of tag_ids)
	model.eval()
	total_loss = 0
	num_words = 0
	for i in range(len(inputs)):
		#print(i)
		example = inputs[i]
		#print(example)
		pred, _ = model(example[1], example[0])
		#print("Model output acquired..")
		# pred is of shape (1, tagset_size, seq_len)
		#print(pred.shape)
		#print(Variable(torch.cuda.LongTensor(np.asarray(outputs[i])).view(1, -1), requires_grad=False).shape)
		loss = criterion(pred, Variable(torch.cuda.LongTensor(np.asarray(outputs[i])).view(1, -1), requires_grad=False))
		#print("Criterion Calculated..")
		loss = torch.sum(loss)
		total_loss += loss.data[0]
		num_words += len(example[1])

	print("Average per sample loss: ", total_loss/len(inputs))
	print("Average per word loss: ", total_loss/num_words)
	model.zero_grad()
		

def eval_accuracy(model, inputs, outputs):
	# inputs is a list of (example: tuple of form :(list of (word: list of char_ids), list of word_ids))
	# outputs is a list of (example: list of tag_ids)
	
	model.eval()
	num_words_correct = 0
	total_words = 0
	per_example_accuracy = 0

	for i in range(len(inputs)):
		example = inputs[i]
		pred, _ = model(example[1], example[0])
		# pred is of shape (1, tagset_size, seq_len)
		_, predicted_tags = torch.max(pred, dim=1)
		# pred is of shape (1, seq_len)
		correct_tags = Variable(torch.cuda.LongTensor(np.asarray(outputs[i])).view(1, -1))
		assert correct_tags.shape==predicted_tags.shape
		
		equal = correct_tags.eq(predicted_tags)
		equals = torch.sum(equal).data[0]
		num_words_correct += equals
		total_words += predicted_tags.shape[0]
		per_example_accuracy += equals/len(example[1])

	per_example_accuracy = per_example_accuracy/len(inputs)
	per_word_accuracy = num_words_correct/total_words
	print("Per Sentence Accuracy: ", per_example_accuracy)
	print("Percentage Words correct: ", per_word_accuracy)
	model.zero_grad()