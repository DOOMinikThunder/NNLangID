# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F



class GRUModel(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional, initial_lr, weight_decay):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lr = initial_lr
        self.weight_decay = weight_decay
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
        self.batch_size = 1
        
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=initial_lr, weight_decay=weight_decay)


    def forward(self, inp, hidden=None):
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = output.view(-1, self.num_classes)
#        for i in range(len(output)):
#            output[i] = F.tanh(output.data[i])
        output = self.log_softmax(output)
        return output, next_hidden


    def initHidden(self):
        return Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
    
    
    def train(self, inputs, targets, eval=False):
        all_pred = 0
        correct_pred = 0
        
        inputs_size = len(inputs)
        num_inputs_minus_one = inputs_size - 1
        for i in range(inputs_size):
            inp = inputs[i]
            target = targets[i]
            hidden = self.initHidden()
            dims = list(inp.size())
            
            output, hidden = self(inp, hidden)
            # transform 3D to 2D tensor for the criterion function
            output = output.view(dims[0], -1)
#           print('OUTPUT:\n', output)
            
            self.zero_grad()
            
            if(eval):
                mean_list = [0]*output.size()[1]
                for pred in output:
                    for i,_ in enumerate(mean_list):
                        mean_list[i] += pred[i].data[0]
                mean_list = [mean/output.size()[0] for mean in mean_list]
                prediction = mean_list.index(max(mean_list))
                if(prediction == int(target.data[0])):
                    correct_pred += 1
                all_pred += 1
            else:
                loss = self.criterion(output, target)
                print('RNN Loss', i, '/', num_inputs_minus_one, ':\n', float(loss.data[0]))
                loss.backward()
                self.optimizer.step()
        if(eval):
#            print('Correct:', correct_pred, '/', all_pred)
            accuracy = correct_pred/all_pred
            return accuracy

#                    loss = criterion(output, target)
#                    print('RNN LOSS:\n', loss)
#                    loss.backward()
#                    optimizer.step()
            
        
    def save_model_checkpoint_to_file(self, state, relative_path_to_file):
        torch.save(state, relative_path_to_file)
        print('Model checkpoint saved to file:', relative_path_to_file)
        
        
    def load_model_checkpoint_from_file(self, relative_path_to_file):
        checkpoint = torch.load(relative_path_to_file)
        start_epoch = checkpoint['start_epoch']
        best_accuracy = checkpoint['best_accuracy']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
#        self.eval()
        print('Model checkpoint loaded from file:', relative_path_to_file)
        return start_epoch, best_accuracy