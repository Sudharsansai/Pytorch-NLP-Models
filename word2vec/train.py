'''
Author: Sudharsansai, UCLA
Training progam for word2vec
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize
from model import CBOW
import helper
import evaluate

CONTEXT_SIZE = 2
train_ratio = 0.8
num_epochs = 2000
print_epochs = 100
batch_size = 8
lr = 1e-3
save_file = 'word2vec_CBOW.pt'
load = False
save = True
EMB_SIZE = 10

data = helper.create_dataset(CONTEXT_SIZE)
train_data, test_data = helper.split_dataset(data, train_ratio)
word_to_ix = helper.create_dictionary()

VOCAB_SIZE = len(word_to_ix)

test_inputs, test_labels = helper.make_batch(test_data, -1)
train_inputs, train_labels = helper.make_batch(train_data, batch_size)

test_inputs = Variable(helper.vectorize_data(test_inputs, word_to_ix), requires_grad=False)
test_labels = Variable(helper.vectorize_data(test_labels, word_to_ix).view(-1), requires_grad=False)

model = CBOW(VOCAB_SIZE, EMB_SIZE)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.NLLLoss()


# Before training
print("TESTING BEFORE TRAINING---------")
model.eval()
print('LOSS: '+str(evaluate.eval_loss(model, test_inputs, test_labels, criterion)))
model.zero_grad()


print('TRAINING STARTS------------------')
model.train()

if(load):
	model.load_state_dict(torch.load(save_file)['model'])
	optimizer.load_state_dict(torch.load(save_file)['optimizer'])

for i in range(num_epochs):
	epoch_loss = 0

	for j in range(len(train_inputs)):
		batch_inp = Variable(helper.vectorize_data(train_inputs[j], word_to_ix), requires_grad=False)
		batch_lab = Variable(helper.vectorize_data(train_labels[j], word_to_ix).view(-1), requires_grad=False)
		y_pred = model(batch_inp)
		loss = criterion(y_pred, batch_lab)
		epoch_loss += loss.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	epoch_loss /= len(train_inputs)
	if(i%print_epochs == 0):
		print('Epoch Number: '+str(i)+" Training Loss: "+str(epoch_loss))

if(save):
	torch.save({
			'model': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, save_file)


# After training
print("TESTING BEFORE TRAINING---------")
model.eval()
print('LOSS: '+str(evaluate.eval_loss(model, test_inputs, test_labels, criterion)))
model.zero_grad()
