'''
Author: Sudharsansai, UCLA
Training program for n-gram LM
'''



import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
from nltk.tokenize import word_tokenize
from model import LSTM_LM
import helper
import evaluate

dtype = torch.cuda.FloatTensor

CONTEXT_SIZE = 4
num_epochs = 20
print_batches = 1000
batch_size = 64
lr = 1e-3
save_file = 'LSTM_LM.pt'
load = False
save = True
EMB_SIZE = 300
HIDDEN_DIM = 128
max_words_for_decoding = 50

train_data, test_data, valid_data = helper.create_dataset(CONTEXT_SIZE)
word_to_ix, idx_to_word = helper.create_dictionary(train_data)

print("Number of Training Examples: ", len(train_data))
print("Number of Test Examples: ", len(test_data))
print("Number of Validation Examples: ", len(valid_data))

VOCAB_SIZE = len(word_to_ix)

test_inputs, test_labels = helper.make_batch(test_data, -1)
train_inputs, train_labels = helper.make_batch(train_data, -1)
valid_inputs, valid_labels = helper.make_batch(valid_data, -1)

test_inputs = Variable(helper.vectorize_data(test_inputs, word_to_ix), requires_grad=False)
test_labels = Variable(helper.vectorize_data(test_labels, word_to_ix).view(-1), requires_grad=False)

train_inputs = helper.vectorize_data(train_inputs, word_to_ix)
train_labels = helper.vectorize_data(train_labels, word_to_ix).view(-1)

valid_inputs = helper.vectorize_data(valid_inputs, word_to_ix)
valid_labels = helper.vectorize_data(valid_labels, word_to_ix).view(-1)

model = LSTM_LM(VOCAB_SIZE, EMB_SIZE, HIDDEN_DIM, batch_size).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss(size_average=True)


# Before training
print("\nTESTING BEFORE TRAINING---------")
model.eval()
print('\nLOSS: '+str(evaluate.eval_loss(model, test_inputs, test_labels, criterion)))
model.zero_grad()

print("\nDECODING BEFORE TRAINING----------")
evaluate.decode(model, word_to_ix["the"], idx_to_word, max_words_for_decoding)


print('\nPERPLEXITY BEFORE TRAINING----------')
print("Test Perplexity: ", evaluate.perplexity(model, test_inputs, test_labels, criterion))


print('\nTRAINING STARTS------------------')
model.train()

if(load):
	model.load_state_dict(torch.load(save_file)['model'])
	optimizer.load_state_dict(torch.load(save_file)['optimizer'])

assert train_inputs.shape[0] == train_labels.shape[0]
print('\nTRAINING DATA SHAPE: ', train_inputs.shape)

for i in range(num_epochs):
	batch_loss = 0
	batch_num = 0
	for j in range(0, train_inputs.shape[0]-batch_size, batch_size):
		batch_inp = Variable(train_inputs[j:j+batch_size][:], requires_grad=False)
		batch_lab = Variable(train_labels[j:j+batch_size].view(-1), requires_grad=False)
		hidden = model.init_hidden(batch_size)
		y_pred, _ = model(batch_inp, hidden)
		loss = criterion(y_pred, batch_lab)
		batch_loss += loss.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if((batch_num%print_batches)==0 and batch_num!=0):
			batch_loss /= print_batches
			print("Epoch Number: ",i," Batch Number: ",batch_num, " Training Loss: ",batch_loss)
			batch_loss = 0

		batch_num+=1

if(save):
	torch.save({
			'model': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, save_file)


# After training
print("\nTESTING BEFORE TRAINING---------")
model.eval()
print('\nLOSS: '+str(evaluate.eval_loss(model, test_inputs, test_labels, criterion)))
model.zero_grad()

print("\nDECODING BEFORE TRAINING----------")
evaluate.decode(model, word_to_ix["the"], idx_to_word, max_words_for_decoding)


print('\nPERPLEXITY BEFORE TRAINING----------')
print("Test Perplexity: ", evaluate.perplexity(model, test_inputs, test_labels, criterion))