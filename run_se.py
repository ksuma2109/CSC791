import torch
# from utils import get_data, prepro
import torchtext
from torchtext import data
from torchtext import datasets
from Baseline import Baseline
# from LSTM import LSTM_model
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
import sys
import pandas as pd
import numpy as np
from torchtext.data import TabularDataset
import string
from torchtext.vocab import Vectors, GloVe, FastText
from torchtext.data import Iterator, BucketIterator
from pandas.api.types import CategoricalDtype

# from spacy.vectors import Vectors
# vectors = Vectors()
# vectors.from_glove("/Users/suma/Desktop/acad/CS791-NLP/hotpot/glove.840B.300d.txt")

print("Start training Semantic Entailment model")
data = pd.read_csv('train.txt', sep=" ", header=None)
data[1], pos_mapping = pd.Series(data[1]).factorize()
pos_mapping = list(pos_mapping)
# print(pos_mapping)
pos_cat = CategoricalDtype(categories=pos_mapping)
# print(pos_cat)
data = data[[0, 1]]
data.to_csv("train_pos.csv", index=False)
# data[2], chunk_mapping = pd.Series(data[2]).factorize()

data_test = pd.read_csv('test.txt', sep=" ", header=None)
data_test[1] = data_test[1].apply(lambda x: pos_mapping.index(x))
data_test = data_test[[0, 1]]
data.to_csv("test_pos.csv", index=False)

# print(data_test.head())

# df = pd.read_csv('../hotpot/glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
# print("Done till here")
# glove = {key: val.values for key, val in df.T.items()}

# print(glove)

# def pos_map(word):
# 	if(pos_mapping.)

def get_data(train_file, test_file):

    tokenize = lambda x: x.split()
    remove_punctuation = lambda x: [w.translate(str.maketrans('', '', string.punctuation)) for w in x]
    pos_map = lambda x: [pos_mapping.index(w) for w in x]
    NUMBER = torchtext.data.Field(dtype=torch.int64)
    TEXT = torchtext.data.ReversibleField(sequential=True, tokenize=tokenize, use_vocab=True, preprocessing=remove_punctuation, lower=True, include_lengths=True, batch_first=True, fix_length=1, pad_token='<pad>', unk_token=' ')
    LABEL = torchtext.data.LabelField(dtype=torch.int64)
    fields = [("word", TEXT), ("pos", LABEL), ("chunk", None)]
    # filtered = filter(lambda p: '\n' == p[0], reader)
    train_data   = TabularDataset(
               path=train_file, # the root directory where the data lies
               format='csv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=fields)
    test_data   = TabularDataset(
               path=test_file, # the root directory where the data lies
               format='csv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=fields)
    TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=300), max_size=400000)
    # TEXT.vocab = torchtext.vocab.GloVe(name='6B', dim=300)
    # TEXT.build_vocab(train_data, vectors=FastText())
    LABEL.build_vocab(train_data)
    # LABEL.build_vocab(test_data)
    # word_embeddings = TEXT.vocab.vectors
    train_iter = BucketIterator(train_data, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)
    test_iter = BucketIterator(test_data, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)
    # print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    # print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size()[1])
    # print ("Label Length: " + str(len(LABEL.vocab)))
    return train_data, test_data, train_iter, test_iter, TEXT, LABEL

# word_mat = torch.FloatTensor(matrix)
# print(word_mat.shape)
train_pos = data[[0]]
target_pos = data[[1]]

train_data, test_data, train_iter, test_iter, TEXT, LABEL = get_data('train_se.csv','test_se.csv' )

# print(train_pos)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch):
	total_epoch_loss = 0
	total_epoch_acc = 0
	# model.cuda()
	optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
	steps = 0
	model.train()
	for idx, batch in enumerate(train_iter):
		# print(idx)
		# print(batch)
		word = batch.word[0]
		pos = batch.pos
		pos = torch.autograd.Variable(pos).long()
		optim.zero_grad()
		if (word.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 32.
		    prediction = model(word, word.size()[0])
		else:
			prediction = model(word)


		prediction = torch.squeeze(prediction, dim=1)
		# print(prediction.size())
		# print(target.size())
		# print(target)
		loss = loss_fn(prediction, pos)
		num_corrects = (torch.max(prediction, 1)[1].view(pos.size()).data == pos.data).float().sum()
		acc = 100.0 * num_corrects/len(batch)
		loss.backward()
		clip_gradient(model, 1e-1)
		optim.step()
		steps += 1

		if steps % 100 == 0:
		    print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

		total_epoch_loss += loss.item()
		total_epoch_acc += acc.item()

	return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
	total_epoch_loss = 0
	total_epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(val_iter):
			word = batch.word[0]
			pos = batch.pos
			pos = torch.autograd.Variable(pos).long()
			if (word.size()[0] is not batch_size):
				prediction = model(word, word.size()[0])
			else:
				prediction = model(word)
			# prediction = model(event1, event2)
			prediction = torch.squeeze(prediction, dim=1)
			loss = loss_fn(prediction, pos)
			num_corrects = (torch.max(prediction, 1)[1].view(pos.size()).data == pos.data).sum()
			acc = 100.0 * num_corrects/len(batch)
			total_epoch_loss += loss.item()
			total_epoch_acc += acc.item()

	return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


learning_rate = 1e-6
batch_size = 64
output_size = len(pos_mapping)
hidden_size = 256
embedding_length = 300
vocab_len = 400001

model = Baseline(batch_size, output_size, hidden_size, vocab_len, embedding_length, TEXT.vocab.vectors)
loss_fn = torch.nn.CrossEntropyLoss(size_average=False, reduction='mean')

for epoch in range(20):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    # val_loss, val_acc = eval_model(model, valid_iter)
    if epoch%1==0:
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        torch.save(model, 'se.pt')

print("Saving model for Semantic Entailment Model ")
test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')



