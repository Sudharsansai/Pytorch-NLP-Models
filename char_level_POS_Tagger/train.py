'''
Author: Sudharsansai, UCLA
Training program for Char Level POS Tagger, which optionally takes a word level component
'''


import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from nltk.tokenize import word_tokenize
from model import POSTagger
import helper
import evaluate

dtype = torch.cuda.FloatTensor

num_epochs = 10
batch_size = 100
lr = 2e-4
save_file = 'POSTagger.pt'
load = False
save = True
word_emb_dim = 128
char_emb_dim = 64
word_hidden_dim = 128
char_hidden_dim = 32
save_per_epochs = 2
threshold = 5

print("\nHYPER-PARAMETERS SETTING:---------------------------\n")
print('Learning rate: ', lr)
print("#Epochs: ", num_epochs)
if(save):
	print("Model to be saved into file: ", save_file)
else:
	print("Model is not going to be saved")
print("Word Embedding Size: ", word_emb_dim)
print("Word-LSTM Hidden dimension: ", word_hidden_dim)
print("Char Embedding Size: ", char_emb_dim)
print("Char-LSTM Hidden dimension: ", char_hidden_dim)
print("Dropout: No Dropout")
print("Frequency Threshold: ", threshold)
print("Without using masks, using ignore_index=0 in NLLLoss")
print("--------------------------------------------------\n")

train_inputs, test_inputs = helper.create_dataset("inputs")
train_tags, test_tags = helper.create_dataset("outputs")

'''
assert len(train_inputs)==len(train_tags)
assert len(test_inputs)==len(test_tags)
'''

print("Dataset Created...")
print("Number of Training Examples: ", len(train_inputs))
print("Number of Test Examples: ", len(test_inputs))

word_to_ix = helper.create_dictionary(train_inputs, threshold)
tags_to_ix = helper.create_dictionary(train_tags)

char_list = []
for sentence in train_inputs:
	for word in sentence:
		char_list.append(list(word))
char_to_ix = helper.create_dictionary(char_list)

word_vocab_size = len(word_to_ix)
char_vocab_size = len(char_to_ix)
tags_vocab_size = len(tags_to_ix)

print("Word Vocabulary Size: ", word_vocab_size)
print("Char Vocabulary Size: ", char_vocab_size)
print("Tags Set Size: ", tags_vocab_size)

train_inputs = helper.vectorize_inputs(train_inputs, word_to_ix, char_to_ix)
test_inputs = helper.vectorize_inputs(test_inputs, word_to_ix, char_to_ix)
# train_inputs and test_inputs are each list of examples
# Each example is a tuple. The first element of each tuple is a list of (word: list of char_ids)
# The second element is a list of word_ids
train_tags = helper.vectorize_tags(train_tags, tags_to_ix)
test_tags = helper.vectorize_tags(test_tags, tags_to_ix)
# Each of train_tags and test_tags is a list of (list of tag ids)

'''
print('Sample example before vectorizing: ------------------\n')
print(train_inputs[0])
print(train_tags[0])
'''

model = POSTagger(tags_vocab_size, char_vocab_size, char_emb_dim, 
	char_hidden_dim, word_vocab_size, word_emb_dim, word_hidden_dim).cuda()
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.NLLLoss(reduce=False)


# Before training
print("\nTesting Before Training on Test Set---------")
model.eval()
evaluate.eval_loss(model, test_inputs, test_tags, criterion)
evaluate.eval_accuracy(model, test_inputs, test_tags)
model.zero_grad()


print('\nTraining Starts------------------')
model.train()

if(load):
	model.load_state_dict(torch.load(save_file)['model'])
	optimizer.load_state_dict(torch.load(save_file)['optimizer'])

for epoch in range(num_epochs):
	batch_loss = 0
	for i in range(len(train_inputs)):
		example = train_inputs[i]
		pred, _ = model(example[1], example[0])
		# pred is of shape (1, tagset_size, seq_len)
		loss = criterion(pred, Variable(torch.cuda.LongTensor(np.asarray(train_tags[i])).view(1, -1), requires_grad=False))
		loss = torch.sum(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_loss += loss.data[0]

		if((i!=0) and (i%batch_size==0)):
			per_example_loss = batch_loss/batch_size
			print("Epoch Number: ", epoch, " Example Number: ", i, " Training Loss per sentence: ", per_example_loss)
			batch_loss = 0
	
	if((epoch!=0) and ((epoch%save_per_epochs)==0)):
		if(save):
			torch.save({
					'model': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
				}, save_file)

if(save):
	torch.save({
			'model': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, save_file)


# After training
print("\nTesting After Training on Test Set---------")
model.eval()
evaluate.eval_loss(model, test_inputs, test_tags, criterion)
evaluate.eval_accuracy(model, test_inputs, test_tags)
model.zero_grad()