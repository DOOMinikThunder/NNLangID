# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from . import BatchGenerator
from evaluation import RNNEvaluator



class GRUModel(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional, initial_lr, weight_decay):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lr = initial_lr
        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=is_bidirectional)
        if (is_bidirectional):
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.output_layer = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.log_softmax = nn.LogSoftmax()
        self.batch_size = 1     # unused dimension
        self.cuda_is_avail = torch.cuda.is_available()
        
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=initial_lr, weight_decay=weight_decay)


    def initHidden(self):
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
        # transfer tensor to GPU if available
        if (self.cuda_is_avail):
            return hidden.cuda()
        else:
            return hidden
    

    def forward(self, inp, hidden=None):
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = output.view(-1, self.num_classes)
#        for i in range(len(output)):
#            output[i] = F.tanh(output.tweet_retriever_data[i])
        output = self.log_softmax(output)
        return output, next_hidden


    def train(self, train_inputs, train_targets, val_inputs, val_targets, batch_size, max_eval_checks_not_improved, max_num_epochs, eval_every_num_batches, lr_decay_every_num_batches, lr_decay_factor, rnn_model_checkpoint_rel_path, system_param_dict):
        batch_generator = BatchGenerator.Batches(train_inputs, train_targets, batch_size)
        num_train_batches_minus_one = batch_generator.num_batches - 1
        max_eval_checks_not_improved_minus_one = max_eval_checks_not_improved - 1
        best_val_mean_loss = float('inf')
        cur_val_mean_loss = float('inf')
        epoch = 0
        total_trained_batches_counter = 0
        eval_checks_not_improved_counter = 0
        continue_training = True
        rnn_evaluator = RNNEvaluator.RNNEvalautor(self)
        # increase the learning rate variable as in the first iteration it will be immediately decreased
        self.lr = self.lr * (1.0 / lr_decay_factor)
        # train until stopping criterium is satisfied or max_num_epochs is reached
        while continue_training and epoch < max_num_epochs:
            for batch_i, (input_batch, target_batch) in enumerate(batch_generator):
                
                if (continue_training):
                    self.zero_grad()
                    chars_loss_acc = 0
                    
                    for tweet_input, tweet_target in zip(input_batch, target_batch):
                        print(input)
                        hidden = self.initHidden()
                        output, hidden = self(tweet_input, hidden)
                        # transform 3D to 2D tensor for the criterion function
                        dims = list(tweet_input.size())
                        output = output.view(dims[0], -1)
        #                print('OUTPUT:\n', output)
                        
                        loss = self.criterion(output, tweet_target)
                        loss.backward()
                        chars_loss_acc += float(loss.data[0])
#                    chars_loss_acc / 
                    print('RNN Loss', batch_i, '/', num_train_batches_minus_one, ': ', float(loss.data[0]))
                    self.optimizer.step()
                    
                    
                    
                    
                     
        
#                  
#                    batch_mean_loss = self.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
#                    if (batch_i % 100 == 0):
#                        print('[EMBEDDING] Epoch', epoch, '| Batch', batch_i, '/', num_train_batched_pairs_minus_one, '| Training mean loss: ', batch_mean_loss.data[0])
#                    batch_mean_loss.backward()
#                    self.optimizer.step()
#                    
#                    # decrease learning rate every lr_decay_every_num_batches
#                    if (total_trained_batches_counter % lr_decay_every_num_batches == 0):
#                        self.lr = self.lr * lr_decay_factor
#                        for param_group in self.optimizer.param_groups:
#                            param_group['lr'] = self.lr
#                    
#                    # evaluate validation set every eval_every_num_batches
#                    if (total_trained_batches_counter % eval_every_num_batches == 0):
#                        cur_val_mean_loss = embedding_evaluator.evaluate_data_set(val_batched_pairs,
#                                                                                  num_neg_samples)
#                        print('========================================')
#                        print('[EMBEDDING] Epoch', epoch, '| Batch', batch_i, '/', num_train_batched_pairs_minus_one, '| Validation mean loss: ', cur_val_mean_loss)
#                        print('========================================')
#            
#                        # check if loss improved, and if so, save embedding weights and model checkpoint to file
#                        # and reset eval_checks_not_improved_counter as model is improving (again)
#                        if (best_val_mean_loss > cur_val_mean_loss):
#                            best_val_mean_loss = cur_val_mean_loss
#                            eval_checks_not_improved_counter = 0
#                            self.save_embed_weights_to_file(embed_weights_rel_path)
#                            self.save_model_checkpoint_to_file({
#                                                        'start_epoch' : epoch + 1,
#                                                        'start_total_trained_batches_counter' : total_trained_batches_counter + 1,
#                                                        'best_val_mean_loss' : best_val_mean_loss,
#                                                        'test_mean_loss' : -1.0,
#                                                        'state_dict': self.state_dict(),
#                                                        'optimizer': self.optimizer.state_dict(),
#                                                        'system_param_dict' : system_param_dict,
#                                                        'vocab_chars' : self.vocab_chars,
#                                                        'vocab_lang' : self.vocab_lang,
#                                                        },
#                                                        embed_model_checkpoint_rel_path)
#                        # as model is not improving: increment counter to stop,
#                        # if counter equals max_eval_checks_not_improved then stop training
#                        else:
#                            print('Not improved evaluation checks:', eval_checks_not_improved_counter, '/', max_eval_checks_not_improved_minus_one)
#                            eval_checks_not_improved_counter += 1
#                            if (eval_checks_not_improved_counter == max_eval_checks_not_improved):
#                                continue_training = False
#                total_trained_batches_counter += 1
#            epoch += 1
            
            
            
        
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