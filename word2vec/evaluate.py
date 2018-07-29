'''
Author: Sudharsansai, UCLA
Evaluation Helper functions for word2vec
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
	pred = model(inputs)
	return criterion(pred, outputs).data[0]
