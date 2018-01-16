import torch
from torch import nn, optim
from torch.autograd import Variable



class SimpleRNN(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn_layer = nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
#                                nonlinearity='relu')
                                nonlinearity='tanh')
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax()
        

    def forward(self, inp, hidden=None):
        output, next_hidden = self.rnn_layer(inp, hidden)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output, next_hidden


    def initHidden(self, num_layers, batch_size, hidden_size):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))



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
    
    
    # initialization
    model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
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
        hidden = model.initHidden(num_layers, batch_size, hidden_size)
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