# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from evaluation import EmbeddingEvaluator



"""
Loss calculation in forward method is based on the Negative sampling objective (NEG) (formula 4)
in paper "Distributed Representations of Words and Phrases and their Compositionality"
by Tomas Mikolov et al. (Oct. 2013).
Initializations and optimizations are suggested by Xiaofei Sun on
https://adoni.github.io/2017/11/08/word2vec-pytorch/ (Access: 11.01.2018)
"""
class SkipGramModel(nn.Module):

    
    def __init__(self, vocab_chars, vocab_lang, embed_dim, initial_lr, sampling_table_min_char_count=1, sampling_table_specified_size_cap=100000000):
        super(SkipGramModel, self).__init__()

        self.vocab_chars = vocab_chars
        self.vocab_lang = vocab_lang
        self.vocab_chars_size = len(vocab_chars)
        self.vocab_lang_size = len(vocab_lang)
        self.embed_dim = embed_dim
        self.lr = initial_lr
        self.embed_hidden = nn.Embedding(self.vocab_chars_size, int(embed_dim), sparse=True)
        self.embed_output = nn.Embedding(self.vocab_chars_size, int(embed_dim), sparse=True)
        self.sampling_table = []
        self.init_embed()
        self.init_sampling_table(vocab_chars, sampling_table_min_char_count, sampling_table_specified_size_cap)
        self.cuda_is_avail = torch.cuda.is_available()
        # no weight_decay and momentum set because they
        # "require the global calculation on embedding matrix, which is extremely time-consuming"
        self.optimizer = optim.SGD(self.parameters(), lr=initial_lr)
        
        
    def init_embed(self):
        init_range = 0.5 / self.embed_dim
        self.embed_hidden.weight.data.uniform_(-init_range, init_range)
        self.embed_output.weight.data.uniform_(-0, 0)


    def init_sampling_table(self, vocab_chars, min_char_count=1, specified_size_cap=100000000):
        char_pow_frequencies = {}
        char_pow_frequencies_acc = 0
        min_char_pow_frequency = float('inf')
        for char in vocab_chars:
            char_pow_frequency = math.pow(vocab_chars[char][1], 0.75)
            char_pow_frequencies_acc = char_pow_frequencies_acc + char_pow_frequency
            char_pow_frequencies[vocab_chars[char][0]] = char_pow_frequency
            # get smallest char_pow_frequency
            if (char_pow_frequency < min_char_pow_frequency):
                min_char_pow_frequency = char_pow_frequency
#        print(char_pow_frequencies)
        # calculate the necessary table_size to have at least min_char_count of each char in the table
        table_specified_size = math.ceil((char_pow_frequencies_acc / min_char_pow_frequency) * min_char_count)
        # cap table size to table_size_cap
        if (table_specified_size > specified_size_cap):
            table_specified_size = specified_size_cap
#        print(table_size)
        # get the number of occurrences of each char in the table (depending on the probability function)
        # and fill the table accordingly
        for char_index in char_pow_frequencies:
            num_of_char = np.round((char_pow_frequencies[char_index] / char_pow_frequencies_acc) * table_specified_size)

            for i in range(int(num_of_char)):
                self.sampling_table.append(char_index)
#        print(self.sampling_table)
                
    
    def get_neg_samples(self, num_pairs, num_samples):
        return np.random.choice(self.sampling_table, size=(num_pairs, num_samples)).tolist()
                

    def forward(self, targets_1_pos, contexts_1_pos, contexts_0_pos_samples):
        losses = []
        # lookup the 1-position weight values for the target char
        # for all target chars in the batch
        targets_1_pos_weights_hidden = self.embed_hidden(targets_1_pos)
        # lookup the 1-position weight values for the context char (backwards from "output layer")
        # for all context chars in the batch
        contexts_1_pos_weights_output = self.embed_output(contexts_1_pos)
        # calculate dot product for each target_1_pos-context_1_pos pair in the batch
        score_contexts_1_pos = torch.mul(targets_1_pos_weights_hidden, contexts_1_pos_weights_output)
        score_contexts_1_pos = torch.sum(score_contexts_1_pos, dim=1)
        # apply log sigmoid function to the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_1_pos = F.logsigmoid(score_contexts_1_pos)
        losses.append(sum(score_contexts_1_pos))
        # use the sampled 0-positions of the context char to lookup the weight values (backwards from "output layer")
        # for all context chars in the batch
        contexts_0_pos_samples_weights_output = self.embed_output(contexts_0_pos_samples)
        # calculate dot product for each target_1_pos-context_0_pos_sample pair
        # for the whole batch
        score_contexts_0_pos_samples = torch.bmm(contexts_0_pos_samples_weights_output, targets_1_pos_weights_hidden.unsqueeze(2))
        score_contexts_0_pos_samples = torch.sum(score_contexts_0_pos_samples, dim=1)
        # apply log sigmoid function to the negative of the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_0_pos_samples = F.logsigmoid(-1 * score_contexts_0_pos_samples)
        losses.append(sum(score_contexts_0_pos_samples))
        # sum up the score_contexts_1_pos and score_contexts_0_pos_samples,
        # make positive (is negative because of log sigmoid function)
        # and normalize the loss by dividing by the batch size
        return (-1 * sum(losses)) / len(targets_1_pos)
    
    
    def train(self, train_batched_pairs, val_batched_pairs, num_neg_samples, max_eval_checks_not_improved, max_num_epochs, eval_every_num_batches, lr_decay_every_num_batches, lr_decay_factor, embed_weights_rel_path, embed_model_checkpoint_rel_path, system_param_dict):
        num_train_batched_pairs_minus_one = len(train_batched_pairs) - 1
        max_eval_checks_not_improved_minus_one = max_eval_checks_not_improved - 1
        best_val_mean_loss = float('inf')
        cur_val_mean_loss = float('inf')
        epoch = 0
        total_trained_batches_counter = 0
        eval_checks_not_improved_counter = 0
        continue_training = True
        embedding_evaluator = EmbeddingEvaluator.EmbeddingEvaluator(self)
        # increase the learning rate variable as in the first iteration it will be immediately decreased
        self.lr = self.lr * (1.0 / lr_decay_factor)
        # train until stopping criterium is satisfied or max_num_epochs is reached
        while continue_training and epoch < max_num_epochs:
            for batch_i, batch in enumerate(train_batched_pairs):
                if (continue_training):
                    batch_size = len(batch)
                    
                    targets_1_pos = [pair[0] for pair in batch]
                    contexts_1_pos = [pair[1] for pair in batch]
                    contexts_0_pos_samples = self.get_neg_samples(batch_size, num_neg_samples)
    
                    targets_1_pos = Variable(torch.LongTensor(targets_1_pos))
                    contexts_1_pos = Variable(torch.LongTensor(contexts_1_pos))
                    contexts_0_pos_samples = Variable(torch.LongTensor(contexts_0_pos_samples))
                    # transfer tensors to GPU if available
                    if (self.cuda_is_avail):
                        targets_1_pos = targets_1_pos.cuda()
                        contexts_1_pos = contexts_1_pos.cuda()
                        contexts_0_pos_samples = contexts_0_pos_samples.cuda()         
        
                    self.optimizer.zero_grad()
                    batch_mean_loss = self.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
                    if (batch_i % 100 == 0):
                        print('[EMBEDDING] Epoch', epoch, '| Batch', batch_i, '/', num_train_batched_pairs_minus_one, '| Training mean loss: ', batch_mean_loss.data[0])
                    batch_mean_loss.backward()
                    self.optimizer.step()
                    
                    # decrease learning rate every lr_decay_every_num_batches
                    if (total_trained_batches_counter % lr_decay_every_num_batches == 0):
                        self.lr = self.lr * lr_decay_factor
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr
                    
                    # evaluate validation set every eval_every_num_batches
                    if (total_trained_batches_counter % eval_every_num_batches == 0):
                        cur_val_mean_loss = embedding_evaluator.evaluate_data_set(val_batched_pairs,
                                                                                  num_neg_samples)
                        print('========================================')
                        print('[EMBEDDING] Epoch', epoch, '| Batch', batch_i, '/', num_train_batched_pairs_minus_one, '| Validation mean loss: ', cur_val_mean_loss)
                        print('========================================')
            
                        # check if loss improved, and if so, save embedding weights and model checkpoint to file
                        # and reset eval_checks_not_improved_counter as model is improving (again)
                        if (best_val_mean_loss > cur_val_mean_loss):
                            best_val_mean_loss = cur_val_mean_loss
                            eval_checks_not_improved_counter = 0
                            self.save_embed_weights_to_file(embed_weights_rel_path)
                            self.save_model_checkpoint_to_file({
                                                        'start_epoch' : epoch + 1,
                                                        'start_total_trained_batches_counter' : total_trained_batches_counter + 1,
                                                        'best_val_mean_loss' : best_val_mean_loss,
                                                        'test_mean_loss' : -1.0,
                                                        'state_dict': self.state_dict(),
                                                        'optimizer': self.optimizer.state_dict(),
                                                        'system_param_dict' : system_param_dict,
                                                        'vocab_chars' : self.vocab_chars,
                                                        'vocab_lang' : self.vocab_lang,
                                                        },
                                                        embed_model_checkpoint_rel_path)
                        # as model is not improving: increment counter to stop,
                        # if counter equals max_eval_checks_not_improved then stop training
                        else:
                            print('Not improved evaluation checks:', eval_checks_not_improved_counter, '/', max_eval_checks_not_improved_minus_one)
                            eval_checks_not_improved_counter += 1
                            if (eval_checks_not_improved_counter == max_eval_checks_not_improved):
                                continue_training = False
                total_trained_batches_counter += 1
            epoch += 1
    
    
    def save_embed_weights_to_file(self, relative_path_to_file):
        # transfer back from GPU to CPU if GPU available
        if (self.cuda_is_avail):
            weights_array = self.embed_hidden.weight.cpu().data.numpy()
        else:
            weights_array = self.embed_hidden.weight.data.numpy()
        writer = open(relative_path_to_file, 'w')
        # write vocabulary size, embedding dimension and number of classes to file
        writer.write('%d %d %d' % (self.vocab_chars_size, self.embed_dim, self.vocab_lang_size))
        # write weights to file (one row for each char of the vocabulary)
        for i in range(self.vocab_chars_size):
            line = ' '.join([str(x) for x in weights_array[i]])
            writer.write('\n%s' % line)
        print('Embedding weights saved to file:', relative_path_to_file)
        
        
    def save_model_checkpoint_to_file(self, state, relative_path_to_file):
        torch.save(state, relative_path_to_file)
        print('Model checkpoint saved to file:', relative_path_to_file)
        
        
    def load_model_checkpoint_from_file(self, relative_path_to_file):
        checkpoint = torch.load(relative_path_to_file)
        start_epoch = checkpoint['start_epoch']
        start_total_trained_batches_counter = checkpoint['start_total_trained_batches_counter']
        best_val_mean_loss = checkpoint['best_val_mean_loss']
        test_mean_loss = checkpoint['test_mean_loss']
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        system_param_dict = checkpoint['system_param_dict']
        vocab_chars = checkpoint['vocab_chars']
        vocab_lang = checkpoint['vocab_lang']
#        self.eval()
        print('Model checkpoint loaded from file:', relative_path_to_file)
        return start_epoch, start_total_trained_batches_counter, best_val_mean_loss, test_mean_loss, system_param_dict, vocab_chars, vocab_lang