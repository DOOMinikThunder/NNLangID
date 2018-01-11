# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import random
#from . import SkipGramModel
from embedding import SkipGramModel
import InputData
#from tqdm import tqdm



class EmbeddingCalculation(object):

    
    def __init__(self):
        self.sampling_table = []
        
    
    # create onehot-vectors for every char and its surrounding context chars (of all tweets)
    # (the last onehot-vector is always the target)
    # (context_window_size = 2 means 2 chars before and after the target char are considered)
    def create_context_target_onehot_vectors(self, context_window_size, tweet_texts_only_embed_chars, chars_for_embed):
        data = []
        total_num_chars_involved = (context_window_size * 2) + 1
        tensor_onehot = torch.FloatTensor(total_num_chars_involved, len(chars_for_embed))
        # for each tweet: create onehot-vector for every char and its context chars
        for tweet in tweet_texts_only_embed_chars:
            for i in range(context_window_size, len(tweet) - context_window_size):
                indexes = []
                # context chars before the target
                for j in range(context_window_size, 0, -1):
                    indexes.append(chars_for_embed[tweet[i - j]][0])
                # context chars after the target
                for j in range(1, context_window_size + 1):
                    indexes.append(chars_for_embed[tweet[i + j]][0])
                    
#    indexes = [chars_for_embed[tweet[i - 2]], chars_for_embed[tweet[i - 1]],
#               chars_for_embed[tweet[i + 1]], chars_for_embed[tweet[i + 2]],
                    
                # the target char
                indexes.append(chars_for_embed[tweet[i]][0])

                tensor = torch.LongTensor(indexes)
                # add 2nd dimension for scatter operation
                tensor.unsqueeze_(1)
      
                # create onehot-vector out of indexes
                tensor_onehot.zero_()
                tensor_onehot.scatter_(1, tensor, 1)
        
#                print(tensor)
#                print(tensor_onehot)
                
                data.append(autograd.Variable(tensor_onehot))
        return data
    
    
    def get_batched_indexed_text(self, tweet_texts_only_embed_chars, chars_for_embed, batch_size):
        batch_tweet_texts = [[]]
        tweet_counter = -1
        char_counter = 0
        batch_counter = 0
        for tweet in tweet_texts_only_embed_chars:
            batch_tweet_texts[batch_counter].append([])
            tweet_counter += 1
            for char in tweet:
                # if batch is full: create new batch list
                if (char_counter == batch_size):
                    batch_counter += 1
                    tweet_counter = 0
                    char_counter = 0
                    batch_tweet_texts.append([])
                    batch_tweet_texts[batch_counter].append([])
                    
                batch_tweet_texts[batch_counter][tweet_counter].append(chars_for_embed[char][0])
                char_counter += 1
        return batch_tweet_texts
    
    
    def get_batched_target_context_index_pairs(self, indexed_tweet_texts, batch_size, max_window_size):
        pairs = [[]]
        pair_counter = 0
        batch_counter = 0
        for tweet_i in range(len(indexed_tweet_texts)):
            for index_j in range(len(indexed_tweet_texts[tweet_i])):
                
                # get random window size (so context chars further away from the target will have lesser weight)
                rnd_window_size = random.randint(1, max_window_size)
                # get pairs in the form (target_index, context_index) for the current window
                for window_k in range(1, rnd_window_size + 1):
                    left_context_index = index_j - window_k
                    right_context_index = index_j + window_k
                    
                    # if not out of bounds to the left
                    if (index_j - window_k >= 0):
                        # if batch is full: create new batch list
                        if (pair_counter == batch_size):
                            pairs.append([])
                            batch_counter += 1
                            pair_counter = 0
                        pairs[batch_counter].append((indexed_tweet_texts[tweet_i][index_j],
                                                     indexed_tweet_texts[tweet_i][left_context_index]))
                        pair_counter += 1

                    # if not out of bounds to the right
                    if (index_j + window_k < len(indexed_tweet_texts[tweet_i])):
                        # if batch is full: create new batch list
                        if (pair_counter == batch_size):
                            pairs.append([])
                            batch_counter += 1
                            pair_counter = 0
                        pairs[batch_counter].append((indexed_tweet_texts[tweet_i][index_j],
                                                     indexed_tweet_texts[tweet_i][right_context_index]))
                        pair_counter += 1
        return pairs
    
    
    def set_sampling_table(self, vocab_chars, min_char_count=1):
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
        table_size = math.ceil((char_pow_frequencies_acc / min_char_pow_frequency) * min_char_count)
#        print(table_size)
        # get the number of occurrences of each char in the table (depending on the probability function)
        # and fill the table accordingly
        for char_index in char_pow_frequencies:
            num_of_char = np.round((char_pow_frequencies[char_index] / char_pow_frequencies_acc) * table_size)

            for i in range(int(num_of_char)):
                self.sampling_table.append(char_index)
#        print(self.sampling_table)
              
        
    def get_neg_samples(self, num_pairs, num_samples):
        return np.random.choice(self.sampling_table, size=(num_pairs, num_samples)).tolist()
            
    
    def calc_embed(self, indexed_tweet_texts, batch_size, vocab_chars, max_context_window_size, num_neg_samples, num_epochs, initial_lr, embed_weights_rel_path, print_testing, sampling_table_min_char_count=1):
        # set embedding dimension to: roundup(log2(vocabulary-size))
        embed_dim = math.ceil(math.log2(len(vocab_chars)))
    #    print(embed_dim)
        
        ##########################################
        # SKIP-GRAM-MODEL WITH NEGATIVE SAMPLING #
        ##########################################
        
        self.set_sampling_table(vocab_chars, sampling_table_min_char_count)
    #    print(embedding_calculation.sampling_table)
    #    print(len(embedding_calculation.sampling_table))
        batched_pairs = self.get_batched_target_context_index_pairs(indexed_tweet_texts, batch_size, max_context_window_size)
#        print(batched_pairs)
        
        skip_gram_model = SkipGramModel.SkipGramModel(len(vocab_chars), embed_dim)
        optimizer = optim.SGD(skip_gram_model.parameters(), lr=initial_lr)
        
        # train skip-gram with negative sampling
        num_epochs_minus_one = num_epochs - 1
        num_batched_pairs_minus_one = len(batched_pairs) - 1
        for epoch_i in range(num_epochs):
            print("Embedding epoch:", epoch_i, "/", num_epochs_minus_one)
            for i, batch in enumerate(batched_pairs):
                targets_1_pos = [pair[0] for pair in batch]
                contexts_1_pos = [pair[1] for pair in batch]
                contexts_0_pos_samples = self.get_neg_samples(len(batch), num_neg_samples)
    #            print(neg_samples)
                
                optimizer.zero_grad()
                loss = skip_gram_model.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
                if (i % 100 == 0):
                    print('Embedding Loss', i, '/', num_batched_pairs_minus_one, ': ', float(loss.data[0]))
                loss.backward()
                optimizer.step()
        
        # write embedding weights to file
        skip_gram_model.save_embed_to_file(embed_weights_rel_path)
    #    print(skip_gram_model.embed_hidden.weight)
        
        
    # TODO: maybe adapt learning rate
        
        
        ###########
        # TESTING #
        ###########
        
        if (print_testing):
#            print("VOCABULARY:\n", vocab_chars)
            input_data = InputData.InputData()
            embed = input_data.create_embed_from_weights_file(embed_weights_rel_path)
            char2index, index2char = input_data.get_char2index_and_index2char(vocab_chars)
            vocab_size = len(vocab_chars)
            
            # get all embed weights
            embed_weights = [embed(Variable(torch.LongTensor([x]))) for x in range(vocab_size)]
#            print(embed_weights)
            
            # print differences between the embeddings
            # (relations on test_embed.csv: g-h, f-e-b-a-c-d)
            print('Embedding vector differences:')
            for i in range(vocab_size):
                char_i = index2char[i]
                for j in range(vocab_size):
#                    if (j >= i):   # print each pair only once
                    char_j = index2char[j]
                    diff = torch.FloatTensor.sum((torch.abs(embed_weights[i] - embed_weights[j]).data[0]))
                    print(char_i, '-', char_j, diff)