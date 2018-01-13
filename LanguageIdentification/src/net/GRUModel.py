# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from . import BatchGenerator



class GRUModel(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional, initial_lr, weight_decay):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax()
        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=is_bidirectional)
        if (is_bidirectional):
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.output_layer = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.batch_size = 1     # unused dimension
        
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=initial_lr, weight_decay=weight_decay)


    def initHidden(self):
        return Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
    

    def forward(self, inp, hidden=None):
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = output.view(-1, self.num_classes)
#        for i in range(len(output)):
#            output[i] = F.tanh(output.data[i])
        output = self.log_softmax(output)
        return output, next_hidden


    def train(self, inputs, targets, batch_size):
        batch_gen = BatchGenerator.Batches(inputs, targets, batch_size)
        num_batches_minus_one = batch_gen.num_batches - 1
        for i, (in_batch, target_batch) in enumerate(batch_gen):
            self.zero_grad()
            #print('in_batch size', len(in_batch))
            for input, target in zip(in_batch, target_batch):
                #print('in size', input.size())
                hidden = self.initHidden()
                #print('in_batch', type(input))
                #print(input)
                output, hidden = self(input, hidden)
                # transform 3D to 2D tensor for the criterion function
                dims = list(input.size())

                output = output.view(dims[0], -1)
#                print('OUTPUT:\n', output)

                loss = self.criterion(output, target)
                #print('rnn Loss', float(loss.data[0]))
                loss.backward()
            print('RNN Loss', i, '/', num_batches_minus_one, ': ', float(loss.data[0]))
            self.optimizer.step()
            
        
    def save_model_checkpoint_to_file(self, state, relative_path_to_file):
        torch.save(state, relative_path_to_file)
        print('Model checkpoint saved to file:', relative_path_to_file)
        
        
    def load_model_checkpoint_from_file(self, relative_path_to_file):
        checkpoint = torch.load(relative_path_to_file)
        start_epoch = checkpoint['start_epoch']
        best_val_accuracy = checkpoint['best_val_accuracy']
        test_accuracy = checkpoint['test_accuracy']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        system_param_dict = checkpoint['system_param_dict']
        vocab_chars = checkpoint['vocab_chars']
        vocab_lang = checkpoint['vocab_lang']
#        self.eval()
        print('Model checkpoint loaded from file:', relative_path_to_file)
        return start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang