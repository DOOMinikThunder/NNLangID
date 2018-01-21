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
from torch import autograd, nn, optim
import torch.nn.functional as F


batch_size = 5
input_size = 4
num_classes = 4
hidden_size = 4
learning_rate = 0.001


torch.manual_seed(42)
input = autograd.Variable(torch.rand(batch_size, input_size)) -0.5
target = autograd.Variable((torch.rand(batch_size)*num_classes).long())
#print('input', input)


class Net(nn.Module):
    """Simple ANN class for testing.
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x


model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
#model.zero_grad() #zero the gradient on the weights in both hidden layers
#model.parameters() #iterator over parameters (weights) over each of the layers
opt = optim.Adam(params=model.parameters(), lr=learning_rate) #params are weights

for  epoch in range(1000):
    out = model(input)
    #print('out', out)
    _, pred = out.max(1)
    print('target', str(target.view(1,-1)).split('\n')[1])
    print('pred', str(pred.view(1,-1)).split('\n')[1])
    loss = F.nll_loss(out, target)
    print('loss', loss.data[0])

    model.zero_grad()
    loss.backward()
    opt.step()


