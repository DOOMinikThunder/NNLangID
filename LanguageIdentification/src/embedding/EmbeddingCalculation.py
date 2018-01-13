# -*- coding: utf-8 -*-

import math
import random
import torch
import torch.autograd as autograd
from torch.autograd import Variable
#from . import SkipGramModel
from embedding import SkipGramModel
import InputData
#from tqdm import tqdm



class EmbeddingCalculation(object):
        
    
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
    
    
    def calc_embed(self, indexed_tweet_texts, batch_size, vocab_chars, vocab_lang, max_context_window_size, num_neg_samples, num_epochs, initial_lr, lr_decay_num_batches, embed_weights_rel_path, print_testing, sampling_table_min_char_count=1, sampling_table_specified_size_cap=100000000):
        # set embedding dimension to: roundup(log2(vocabulary-size))
        embed_dim = math.ceil(math.log2(len(vocab_chars)))
    #    print(embed_dim)
        
        ##########################################
        # SKIP-GRAM-MODEL WITH NEGATIVE SAMPLING #
        ##########################################
        
        batched_pairs = self.get_batched_target_context_index_pairs(indexed_tweet_texts, batch_size, max_context_window_size)
#        print(batched_pairs)
        skip_gram_model = SkipGramModel.SkipGramModel(vocab_chars=vocab_chars,
                                                      vocab_lang=vocab_lang,
                                                      embed_dim=embed_dim,
                                                      initial_lr=initial_lr,
                                                      sampling_table_min_char_count=sampling_table_min_char_count,
                                                      sampling_table_specified_size_cap=sampling_table_specified_size_cap)
        
        # train skip-gram with negative sampling
#        skip_gram_model.scheduler.step()
        skip_gram_model.train(batched_pairs=batched_pairs,
                              num_neg_samples=num_neg_samples,
                              num_epochs=num_epochs,
                              lr_decay_num_batches=lr_decay_num_batches)
        
        # write embedding weights to file
        skip_gram_model.save_embed_to_file(embed_weights_rel_path)
    #    print(skip_gram_model.embed_hidden.weight)
        
        
        ###########
        # TESTING #
        ###########
        
        if (print_testing):
#            print("VOCABULARY:\n", vocab_chars)
            input_data = InputData.InputData()
            embed, num_classes = input_data.create_embed_from_weights_file(embed_weights_rel_path)
            char2index, index2char = input_data.get_char2index_and_index2char(vocab_chars)
            vocab_size = len(vocab_chars)
            
            # get all embed weights
            embed_weights = [embed(Variable(torch.LongTensor([x]))) for x in range(vocab_size)]
#            print(embed_weights)
            
            # print differences between the embeddings
            # (relations on test_embed.csv: g-h, f-e-b-a-c-d)
            print('Embedding vector differences (relations on test_embed.csv: g-h, f-e-b-a-c-d):')
            for i in range(vocab_size):
                char_i = index2char[i]
                for j in range(vocab_size):
#                    if (j >= i):   # print each pair only once
                    char_j = index2char[j]
                    diff = torch.FloatTensor.sum((torch.abs(embed_weights[i] - embed_weights[j]).data[0]))
                    print(char_i, '-', char_j, diff)