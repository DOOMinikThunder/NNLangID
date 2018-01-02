# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable



class GRUModel(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.log_softmax = nn.LogSoftmax()
        if (is_bidirectional):
            self.num_directions = 2
            self.gru_layer = nn.GRU(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=True)
            # hidden_size * 2 because of bidirectional
            self.output_layer = nn.Linear(hidden_size * self.num_directions, num_classes)
        else:
            self.num_directions = 1
            self.gru_layer = nn.GRU(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=False)
            self.output_layer = nn.Linear(hidden_size, num_classes)
            

    def forward(self, inp, hidden=None):
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output, next_hidden


    def initHidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
    
    
    def train(self, inputs, targets, batch_size, num_batches, num_epochs):
        criterion = torch.nn.NLLLoss()
        optimizer = optim.Adam(params=self.parameters())
        
        num_epochs_minus_one = num_epochs - 1
        num_batches_minus_one = num_batches - 1
        for epoch in range(num_epochs):
            for batch in range(num_batches):
                print('RNN epoch:', epoch, '/', num_epochs_minus_one, '\nRNN batch:', batch, '/', num_batches_minus_one)
                    
                for i in range(len(inputs)):
                    inp = inputs[i]
                    target = targets[i]
                    hidden = self.initHidden(batch_size)
                    dims = list(inp.size())
                    
                    output, hidden = self(inp, hidden)
                    # transform 3D to 2D tensor for the criterion function
                    output = output.view(dims[0], -1)
#                    print('OUTPUT:\n', output)
                    
                    self.zero_grad()
                    loss = criterion(output, target)
                    print('RNN LOSS:\n', loss)
                    loss.backward()
                    optimizer.step()