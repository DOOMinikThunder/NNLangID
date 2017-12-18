
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

"""
ignore this file for the moment
rnn still not working
"""
class BiGRU(nn.Module):
	def __init__(self, input_size, hidden_layers, hidden_size, output_size):
		super(BiGRU, self).__init__()

		self.classes = 2
		self.input_layer = nn.ReLU()
		self.rnn_layer = nn.GRU(input_size=input_size,
								hidden_size=hidden_size,
								num_layers=hidden_layers,
								bidirectional=True
								)
		self.output_layer = nn.Linear(in_features=hidden_size,
									  out_features=self.classes)

	def forward(self, inputs):
		#input times weight
		#add a bias
		#activate
		#thx siraj
		pass

