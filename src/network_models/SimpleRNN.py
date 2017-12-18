import torch
from torch import nn, autograd, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math, random

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super().__init__()
		self.hidden_size = hidden_size
		self.in_layer = nn.Linear(input_size, hidden_size)
		#self.embedding = nn.Embedding(input_size, hidden_size)
		self.rnn_layer = nn.RNN(input_size=hidden_size,
								hidden_size=hidden_size,
								num_layers=num_layers,
								nonlinearity='tanh')
		self.out_layer = nn.Linear(hidden_size, input_size)
		#self.decoder = nn.Linear(hidden_size, input_size)
		#self.out_layer = nn.Linear(hidden_size[1], num_classes)

	def forward(self, x, hidden=None):
		#x = self.embedding(x)
		x = self.in_layer(x)
		output, hidden = self.rnn_layer(x, hidden)
		output = self.out_layer(output)
		#decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
		#return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

in_size = 3
h_size = 4
num_layers = 2
model = RNN(input_size=in_size, hidden_size=h_size, num_layers=num_layers)
optimizer = optim.Adam(params=model.parameters())

seq_len = 4
batch_size = 20

torch.manual_seed(99)
input = autograd.Variable(torch.rand(seq_len, batch_size, in_size))
target = autograd.Variable(torch.rand(batch_size, in_size))
#print('input', input)
#print('target', target)
model.initHidden()
criterion = torch.nn.NLLLoss()
epochs = 20
for epoch in range(epochs):
	out, hidden = model(input)
	print('out', out)
	#loss = F.nll_loss(out, target)
	loss = criterion(out, target)
	model.zero_grad()
	loss.backward()
	optimizer.step()

