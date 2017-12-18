import torch
from torch import nn, autograd, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math, random

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super().__init__()
		#self.in_layer = nn.Linear(input_size, hidden_size[0])
		#self.embedding = nn.Embedding(input_size, hidden_size)
		self.rnn_layer = nn.RNN(input_size=input_size,
								hidden_size=hidden_size,
								num_layers=num_layers,
								nonlinearity='tanh')
		#self.decoder = nn.Linear(hidden_size, input_size)
		#self.out_layer = nn.Linear(hidden_size[1], num_classes)

	def forward(self, x, hidden=None):
		#x = self.embedding(x)
		output, hidden = self.rnn_layer(x, hidden)
		#decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
		#return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
		return output, hidden

batch_size = 2
seq_len = 20

input_size = 5
hidden_size = 4
num_layers = 3

torch.manual_seed(42)
example = [i for i in range(0,seq_len)]
input_ex = autograd.Variable(torch.rand(seq_len,input_size))
target_ex = autograd.Variable(torch.rand(seq_len,2))
print('input_ex', input_ex)
print('target_ex', target_ex)

model = nn.RNN(input_size=input_size,
								hidden_size=hidden_size,
								num_layers=num_layers,
								nonlinearity='tanh')
opt = optim.Adam(params=model.parameters())

num_epochs = 50
for epoch in range(0, num_epochs):
	out, state = model(input_ex)
	print(out)
	loss = F.nll_loss(out, target_ex)

	model.zero_grad()
	loss.backwards()
	opt.step()


