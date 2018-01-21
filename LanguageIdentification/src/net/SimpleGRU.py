# -*- coding: utf-8 -*-

#    MIT License
#    
#    Copyright (c) 2018 Alexander Heilig, Dominik Sauter, Tabea Kiupel
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.


import torch
from torch import nn, optim
from torch.autograd import Variable


class SimpleGRU(nn.Module):
    """Simple GRU model class for testing.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleGRU, self).__init__()
        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=True)
        # hidden_size * 2 because of bidirectional
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax()
        
    def forward(self, inp, hidden=None):
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output, next_hidden

    def initHidden(self, num_layers, num_directions, batch_size, hidden_size):
        return Variable(torch.zeros(num_layers * num_directions, batch_size, hidden_size))


def main():
    
    # parameters
    input_size = 5
    hidden_size = 4
    num_layers = 1
    seq_len = 4
    batch_size = 1
    num_classes = 2
    num_epochs = 1000
    num_batches = 1
    num_directions = 2
    

    # initialization
    model = SimpleGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters())
    
    inp = Variable(torch.FloatTensor([[[1, 2, 3, 4, 5]],
                                      [[5, 4, 3, 2, 1]],
                                      [[1, 2, 3, 4, 5]],
                                      [[5, 4, 3, 2, 1]]]))
    target = Variable(torch.LongTensor([0, 1, 0, 1]))
    print('INPUT:\n', inp)
    print('TARGET:\n', target)
    print('MODEL:\n', model)
    
    # training
    num_epochs_minus_one = num_epochs - 1
    num_batches_minus_one = num_batches - 1
    for epoch in range(num_epochs):
        hidden = model.initHidden(num_layers, num_directions, batch_size, hidden_size)
        for batch in range(num_batches):
            print('Epoch:', epoch, '/', num_epochs_minus_one, '\nBatch:', batch, '/', num_batches_minus_one)
            output, hidden = model(inp, hidden)
            # transform 3D to 2D tensor for the criterion function
            output = output.view(seq_len, -1)
            print('OUTPUT:\n', output)
            
            model.zero_grad()
            loss = criterion(output, target)
            print('LOSS:\n', loss)
            loss.backward()
            optimizer.step()
        
        
if __name__ == '__main__':
    main()