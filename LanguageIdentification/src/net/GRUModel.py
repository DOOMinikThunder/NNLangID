# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from . import BatchGenerator
from evaluation import RNNEvaluator



class GRUModel(nn.Module):
    
    
    def __init__(self, vocab_chars, vocab_lang, input_size, num_classes, system_param_dict):
        """

        Args:
        	vocab_chars: every character occurence as a dict of character: index, occurrences
        	vocab_lang: every language occurence as a dict of language: index, occurences
        	input_size: input size of of character embeddings
        	num_classes: number of languages
        	system_param_dict: contains hyperparameters
        """
        super(GRUModel, self).__init__()
        self.vocab_chars = vocab_chars
        self.vocab_lang = vocab_lang
        self.input_size = input_size
        self.num_classes = num_classes
        self.system_param_dict = system_param_dict
        self.hidden_size=system_param_dict['hidden_size_rnn']
        self.num_layers=system_param_dict['num_layers_rnn']
        self.is_bidirectional=system_param_dict['is_bidirectional']
        self.lr = system_param_dict['initial_lr_rnn']
        self.weight_decay=system_param_dict['weight_decay_rnn']
        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                bidirectional=self.is_bidirectional)
        if (self.is_bidirectional):
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.output_layer = nn.Linear(self.hidden_size * self.num_directions, num_classes)
        self.log_softmax = nn.LogSoftmax()
        self.batch_size = 1     # unused dimension
        self.cuda_is_avail = system_param_dict['cuda_is_avail']
        
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def initHidden(self):
        """
        before forwarding a new set of data, the initial rnn state can be set with this method
        Returns: rnn state

        """
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
        # transfer tensor to GPU if available
        if (self.cuda_is_avail):
            return hidden.cuda()
        else:
            return hidden
    

    def forward(self, inp, hidden=None):
        """
        forward propagation
        Args:
        	inp: one tensor of input size, i.e. one tweet
        	hidden: the previous hidden rnn state

        Returns:
        	output: prediction for the input
        	next_hidden: the new hidden state

        """
        output, next_hidden = self.gru_layer(inp, hidden)
        output = self.output_layer(output)
        output = output.view(-1, self.num_classes)
#        for i in range(len(output)):
#            output[i] = F.tanh(output.tweet_retriever_data[i])
        output = self.log_softmax(output)
        return output, next_hidden


    def train(self, train_inputs, train_targets, val_inputs, val_targets):
        """
        model's training method
        iterates over epochs and batches
        updates weights after each batch, saves the best model and decays learning rate
        Args:
        	train_inputs: set of all tweets to be trained
        	train_targets: set of all targets for input tweets
        	val_inputs: set all tweets to be used for validation
        	val_targets: set of all targets for validation tweets

        Returns:

        """
        batch_size = self.system_param_dict['batch_size_rnn']
        max_eval_checks_not_improved = self.system_param_dict['max_eval_checks_not_improved_rnn']
        max_num_epochs = self.system_param_dict['max_num_epochs_rnn']
        eval_every_num_batches = self.system_param_dict['eval_every_num_batches_rnn']
        lr_decay_factor = self.system_param_dict['lr_decay_factor_rnn']
        rnn_model_checkpoint_rel_path = self.system_param_dict['rnn_model_checkpoint_rel_path']
        batch_generator = BatchGenerator.Batches(train_inputs, train_targets, batch_size)
        num_train_batches_minus_one = batch_generator.num_batches - 1
        max_eval_checks_not_improved_minus_one = max_eval_checks_not_improved - 1
        best_val_mean_loss = float('inf')
        cur_val_mean_loss = float('inf')
        best_val_accuracy = -1.0
        cur_val_accuracy = -1.0
        epoch = 0
        total_trained_batches_counter = 0
        eval_checks_not_improved_counter = 0
        max_eval_checks_not_improved_half = max_eval_checks_not_improved / 2
        continue_training = True
        rnn_evaluator = RNNEvaluator.RNNEvaluator(self)
        # increase the learning rate variable as in the first iteration it will be immediately decreased
        self.lr = self.lr * (1.0 / lr_decay_factor)
        # train until stopping criterium is satisfied or max_num_epochs is reached
        while continue_training and epoch < max_num_epochs:
            for batch_i, (input_batch, target_batch) in enumerate(batch_generator):
                if (continue_training):  
                    self.zero_grad()
                    batch_loss_acc = 0
                    # for every tweet in the batch
                    for tweet_input, tweet_target in zip(input_batch, target_batch):
                        hidden = self.initHidden()
                        output, hidden = self(tweet_input, hidden)
                        # transform 3D to 2D tensor for the criterion function
                        dims = list(tweet_input.size())
                        output = output.view(dims[0], -1)
                        
                        loss = self.criterion(output, tweet_target)
                        loss.backward()
                        batch_loss_acc += loss.data[0]
                    batch_mean_loss = batch_loss_acc / batch_size
                    if (batch_i % 10 == 0):
                        print('[RNN] Epoch', epoch, '| Batch', batch_i, '/', num_train_batches_minus_one, '| Training mean loss: ', batch_mean_loss)
                    self.optimizer.step()
                    
                    # evaluate validation set every eval_every_num_batches
                    if (total_trained_batches_counter % eval_every_num_batches == 0):
                        cur_val_mean_loss, cur_val_accuracy = rnn_evaluator.evaluate_data_set(val_inputs,
                                                                                              val_targets,
                                                                                              n_highest_probs=1)
                        print('========================================')
                        print('[RNN] Epoch', epoch, '| Batch', batch_i, '/', num_train_batches_minus_one, '| Validation mean loss: ', cur_val_mean_loss)
                        print('========================================')
                        print('[RNN] Epoch', epoch, '| Batch', batch_i, '/', num_train_batches_minus_one, '| Validation accuracy: ', cur_val_accuracy)
                        print('========================================')
                        
                        # check if accuracy is better than the best accuracy observed so far
                        if (best_val_accuracy < cur_val_accuracy):
                            best_val_accuracy = cur_val_accuracy
                            
                        # check if loss improved, and if so, save model checkpoint to file
                        # and reset eval_checks_not_improved_counter as model is improving (again)
                        if (best_val_mean_loss > cur_val_mean_loss):
                            best_val_mean_loss = cur_val_mean_loss
                            eval_checks_not_improved_counter = 0
                            self.save_model_checkpoint_to_file({
                                                                'system_param_dict': self.system_param_dict,
                                                                'results_dict': {
                                                                                'start_epoch': epoch + 1,
                                                                                'start_total_trained_batches_counter': total_trained_batches_counter + 1,
                                                                                'best_val_mean_loss': best_val_mean_loss,
                                                                                'best_val_accuracy': best_val_accuracy,
                                                                                'state_dict': self.state_dict(),
                                                                                'optimizer': self.optimizer.state_dict(),
                                                                                'vocab_chars': self.vocab_chars,
                                                                                'vocab_lang': self.vocab_lang,
                                                                                },
                                                                },
                                                                rnn_model_checkpoint_rel_path)
                        # as model is not improving: increment counter to stop, and eventually start decreasing the learning rate;
                        # if counter equals max_eval_checks_not_improved then stop training
                        else:
                            print('Not improved evaluation checks:', eval_checks_not_improved_counter, '/', max_eval_checks_not_improved_minus_one)
                            eval_checks_not_improved_counter += 1
                            # when half of maximal not improved eval checks is reached:
                            # decrease learning rate every eval check as long as there is no improvement
                            if (eval_checks_not_improved_counter >= max_eval_checks_not_improved_half):
                                self.lr = self.lr * lr_decay_factor
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = self.lr
                                print('Learning rate decreased to:', self.lr)
                            # stop training when maximum of not improved eval checks is reached
                            if (eval_checks_not_improved_counter == max_eval_checks_not_improved):
                                continue_training = False
                total_trained_batches_counter += 1
            epoch += 1
            
        
    def save_model_checkpoint_to_file(self, state, relative_path_to_file):
        """

        Args:
        	state:
        	relative_path_to_file:

        Returns:

        """
        torch.save(state, relative_path_to_file)
        print('Model checkpoint saved to file:', relative_path_to_file)
        
        
    def load_model_checkpoint_from_file(self, relative_path_to_file):
        """

        Args:
        	relative_path_to_file:

        Returns:

        """
        state = torch.load(relative_path_to_file)
        results_dict = state['results_dict']
        self.load_state_dict(results_dict['state_dict'])
        self.optimizer.load_state_dict(results_dict['optimizer'])
        self.vocab_chars = results_dict['vocab_chars']
        self.vocab_lang = results_dict['vocab_lang']
#        self.eval()
        print('Model checkpoint loaded from file:', relative_path_to_file)
        return state