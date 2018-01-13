# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



"""
Loss calculation in forward method is based on the Negative sampling objective (NEG) (formula 4)
in paper "Distributed Representations of Words and Phrases and their Compositionality"
by Tomas Mikolov et al. (Oct. 2013).
Initializations and optimizations are suggested by Xiaofei Sun on
https://adoni.github.io/2017/11/08/word2vec-pytorch/ (Access: 11.01.2018)
"""
class SkipGramModel(nn.Module):

    
    def __init__(self, vocab_chars, vocab_lang, embed_dim, initial_lr, sampling_table_min_char_count=1, sampling_table_specified_size_cap=100000000):
        super().__init__()
        self.vocab_chars_size = len(vocab_chars)
        self.vocab_lang_size = len(vocab_lang)
        self.embed_dim = embed_dim
        self.initial_lr = initial_lr
        self.embed_hidden = nn.Embedding(self.vocab_chars_size, embed_dim, sparse=True)
        self.embed_output = nn.Embedding(self.vocab_chars_size, embed_dim, sparse=True)
        self.sampling_table = []
        self.init_embed()
        self.init_sampling_table(vocab_chars, sampling_table_min_char_count, sampling_table_specified_size_cap)
#        print(self.sampling_table)
#        print(len(self.sampling_table))
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
        min_char_pow_frequency = math.inf
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
        targets_1_pos_weights_hidden = self.embed_hidden(autograd.Variable(torch.LongTensor(targets_1_pos)))
#        print(targets_1_pos_weights_hidden)
        # lookup the 1-position weight values for the context char (backwards from "output layer")
        # for all context chars in the batch
        contexts_1_pos_weights_output = self.embed_output(autograd.Variable(torch.LongTensor(contexts_1_pos)))
#        print(contexts_1_pos_weights_output)
        # calculate dot product for each target_1_pos-context_1_pos pair in the batch
        score_contexts_1_pos = torch.mul(targets_1_pos_weights_hidden, contexts_1_pos_weights_output)
#        print(score_contexts_1_pos)
        score_contexts_1_pos = torch.sum(score_contexts_1_pos, dim=1)
#        print(score_contexts_1_pos)
        # apply log sigmoid function to the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_1_pos = F.logsigmoid(score_contexts_1_pos)
#        print(score_contexts_1_pos)
        losses.append(sum(score_contexts_1_pos))
        # use the sampled 0-positions of the context char to lookup the weight values (backwards from "output layer")
        # for all context chars in the batch
        contexts_0_pos_samples_weights_output = self.embed_output(autograd.Variable(torch.LongTensor(contexts_0_pos_samples)))
#        print(contexts_0_pos_samples_weights_output)
        # calculate dot product for each target_1_pos-context_0_pos_sample pair
        # for the whole batch
        score_contexts_0_pos_samples = torch.bmm(contexts_0_pos_samples_weights_output, targets_1_pos_weights_hidden.unsqueeze(2))
#        print(score_contexts_0_pos_samples)
        score_contexts_0_pos_samples = torch.sum(score_contexts_0_pos_samples, dim=1)
#        print(score_contexts_0_pos_samples)
        # apply log sigmoid function to the negative of the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_0_pos_samples = F.logsigmoid(-1 * score_contexts_0_pos_samples)
#        print(score_contexts_0_pos_samples)
        losses.append(sum(score_contexts_0_pos_samples))
        # sum up the score_contexts_1_pos and score_contexts_0_pos_samples,
        # make positive (is negative because of log sigmoid function)
        # and normalize the loss by dividing by the batch size
#        print(losses)
        return (-1 * sum(losses)) / len(targets_1_pos)
    
    
    def train(self, batched_pairs, num_neg_samples, num_epochs, lr_decay_num_batches):
        num_batched_pairs = len(batched_pairs)
        num_batched_pairs_minus_one = num_batched_pairs - 1
        num_epochs_minus_one = num_epochs - 1
        total_batch_counter = 0
        total_batch_counter_max = num_epochs * num_batched_pairs
        for epoch_i in range(num_epochs):
            for batch_j, batch in enumerate(batched_pairs):
                targets_1_pos = [pair[0] for pair in batch]
                contexts_1_pos = [pair[1] for pair in batch]
                contexts_0_pos_samples = self.get_neg_samples(len(batch), num_neg_samples)
    #            print(neg_samples)
                total_batch_counter = (epoch_i * num_batched_pairs) + batch_j            
    
                self.optimizer.zero_grad()
                loss = self.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
                if (batch_j % 100 == 0):
                    print('[EMBEDDING] Epoch', epoch_i, '/', num_epochs_minus_one, '| Batch', batch_j, '/', num_batched_pairs_minus_one, '| Loss: ', float(loss.data[0]))
                loss.backward()
                self.optimizer.step()
                
                # decrease learning rate every lr_decay_num_batches
                if (total_batch_counter % lr_decay_num_batches == 0):
                    updated_lr = self.initial_lr * (1.0 - 1.0 * total_batch_counter / total_batch_counter_max)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = updated_lr
    
    
    def save_embed_to_file(self, relative_path_to_file):
        weights_array = self.embed_hidden.weight.data.numpy()
        writer = open(relative_path_to_file, 'w')
        # write vocabulary size and embedding dimension to file
        writer.write('%d %d %d' % (self.vocab_chars_size, self.embed_dim, self.vocab_lang_size))
        # write weights to file (one row for each char of the vocabulary)
        for i in range(self.vocab_chars_size):
            line = ' '.join([str(x) for x in weights_array[i]])
            writer.write('\n%s' % line)
        print('Embedding weights saved to file:', relative_path_to_file)