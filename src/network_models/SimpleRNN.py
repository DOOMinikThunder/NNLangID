import torch
from torch import nn, autograd, optim
from torch.autograd import Variable
#import torch.nn.functional as F



class RNN(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
#        self.in_layer = nn.Linear(input_size, hidden_size)
        self.rnn_layer = nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
#                                nonlinearity='relu')
                                nonlinearity='tanh')
#        self.out_layer = nn.Linear(hidden_size, input_size)
        self.log_softmax = nn.LogSoftmax()
        
        
        #self.embedding = nn.Embedding(input_size, hidden_size)
        #self.decoder = nn.Linear(hidden_size, input_size)
        #self.out_layer = nn.Linear(hidden_size[1], num_classes)


    def forward(self, inp, hidden=None):
#        output = self.in_layer(inp)
        output, next_hidden = self.rnn_layer(inp, hidden)
        output = self.log_softmax(output)
#        output = self.out_layer(output)
        

        #x = self.embedding(x)
        #decoded = s#x = self.embedding(xelf.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        
        return output, next_hidden


    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))




input_size = 3
hidden_size = 2
num_layers = 1

model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
criterion = torch.nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())

seq_len = 1
batch_size = 4

torch.manual_seed(99)
#inp = autograd.Variable(torch.rand(seq_len, batch_size, input_size))
#inp = autograd.Variable(torch.rand(batch_size, input_size))
inp = autograd.Variable(torch.randn(seq_len, batch_size, input_size), requires_grad=True)
hidden = model.initHidden()
target = autograd.Variable(torch.LongTensor([1, 0, 1, 0]))
#target = autograd.Variable(torch.rand(batch_size, hidden_size))
print('INPUT\n', inp)
print('HIDDEN\n', hidden)
print('TARGET\n', target)
print(model)

num_classes = 2
            
num_epochs = 3
num_epochs_minus_one = num_epochs - 1
for epoch in range(num_epochs):
    print('Epoch:', epoch, '/', num_epochs_minus_one)
    output, next_hidden = model(inp, hidden)
#    print('OUTPUT\n', output)
    output = output.view(batch_size, num_classes)
    print('VIEW\n', output)
#    print('NEXT_HIDDEN\n', next_hidden)
    model.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


    #loss = F.nll_loss(out, target)
