# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import unicodecsv as csv
import os
import numpy as np
import math



class EmbeddingCalculation(object):

    
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        
    
    def fetch_tweet_texts_from_file(self, relative_path_to_file):
        tweet_texts = []
        with open(relative_path_to_file, 'rb') as file:
            reader =  csv.reader(file, delimiter=';', encoding='utf-8')
    
#            i = 0
            
            # skip first row (['\ufeff'])
            next(reader)
            for row in reader:
    
#                if(i > 10):
#                    break
    
#                print(i)
                tweet_texts.append(row[1])
#                print(tweet_texts)
#                i = i + 1
            return tweet_texts
        
        
    def get_chars_occurring_min_x_times(self, tweet_texts, x):
        occurred_chars = {}
        # count occurrence of each char in the entire corpus
        for tweet in tweet_texts:
            for char in tweet:
                # if already occurred, increment counter by 1
                if (char in occurred_chars):
                    occurred_chars[char] = occurred_chars[char] + 1
                # if occurred for the first time, add to dict with count 1
                else:
                    occurred_chars[char] = 1
                    
        chars_for_embed = {}
        counter = 0
        # fill new dict with the chars that occurred at least x times
        # (filled values are tuples: (onehot-index, number of occurrences))
        for char in occurred_chars:
            if (occurred_chars[char] >= x):
                chars_for_embed[char] = (counter, occurred_chars[char])
                counter = counter + 1
        return chars_for_embed
    

    def get_only_embed_chars(self, tweet_texts, chars_for_embed):
        tweet_texts_only_embed_chars = []
        for i in range(len(tweet_texts)):
            tweet_texts_only_embed_chars.append([])
            for j in range(len(tweet_texts[i])):
                if (tweet_texts[i][j] in chars_for_embed):
                    tweet_texts_only_embed_chars[i].append(tweet_texts[i][j])
        return tweet_texts_only_embed_chars
    
    
    def set_char2index_and_index2char(self, chars_for_embed):
        for char in chars_for_embed:
            self.char2index[char] = chars_for_embed[char][0]
            self.index2char[chars_for_embed[char][0]] = char
            
    
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
    
    
    def get_batched_indexed_text(self, tweet_texts_only_embed_chars, batch_size):
        batch_tweet_texts = [[]]
        tweet_counter = -1
        char_counter = 0
        batch_counter = 0
        for tweet in tweet_texts_only_embed_chars:
            batch_tweet_texts[batch_counter].append([])
            tweet_counter = tweet_counter + 1
            for char in tweet:
                # if batch is full: create new batch list
                if (char_counter == batch_size):
                    batch_counter = batch_counter + 1
                    tweet_counter = 0
                    char_counter = 0
                    batch_tweet_texts.append([])
                    batch_tweet_texts[batch_counter].append([])
                    
                batch_tweet_texts[batch_counter][tweet_counter].append(self.char2index[char])
                char_counter = char_counter + 1
        return batch_tweet_texts
    
    
    def get_indexed_tweet_texts(self, tweet_texts_only_embed_chars):
        indexed_tweet_texts = []
        for i in range(len(tweet_texts_only_embed_chars)):
            indexed_tweet_texts.append([])
            for j in range(len(tweet_texts_only_embed_chars[i])):
                indexed_tweet_texts[i].append(self.char2index[tweet_texts_only_embed_chars[i][j]])
        return indexed_tweet_texts
    
    
    def get_batched_target_context_index_pairs(self, indexed_tweet_texts, batch_size, window_size):
        pairs = [[]]
        pair_counter = 0
        batch_counter = 0
        for tweet_i in range(len(indexed_tweet_texts)):
            for index_j in range(len(indexed_tweet_texts[tweet_i])):
                # if batch is full: create new batch list
                if (pair_counter == batch_size):
                    pairs.append([])
                    batch_counter = batch_counter + 1
                    pair_counter = 0

                # get pairs in the form (target_index, context_index) for the current window
                for window_k in range(1, window_size + 1):
                    left_context_index = index_j - window_k
                    right_context_index = index_j + window_k
                    
                    # if not out of bounds to the left
                    if (index_j - window_k >= 0):
                        pairs[batch_counter].append((indexed_tweet_texts[tweet_i][index_j],
                                                     indexed_tweet_texts[tweet_i][left_context_index]))
                        pair_counter = pair_counter + 1
                        # if batch is full: create new batch list
                        if (pair_counter == batch_size):
                            pairs.append([])
                            batch_counter = batch_counter + 1
                            pair_counter = 0
                            
                    # if not out of bounds to the right
                    if (index_j + window_k < len(indexed_tweet_texts[tweet_i])):
                        pairs[batch_counter].append((indexed_tweet_texts[tweet_i][index_j],
                                                     indexed_tweet_texts[tweet_i][right_context_index]))
                        pair_counter = pair_counter + 1
                        # if batch is full: create new batch list
                        if (pair_counter == batch_size):
                            pairs.append([])
                            batch_counter = batch_counter + 1
                            pair_counter = 0
        return pairs
        
    
    
def main():
    
#    rel_path = "../data/uniformly_sampled_dl.csv"
    rel_path = "../data/test.csv"
    ec = EmbeddingCalculation()
    
    tweet_texts = ec.fetch_tweet_texts_from_file(rel_path)
    print(tweet_texts)
    chars_for_embed = ec.get_chars_occurring_min_x_times(tweet_texts, 2)
    print(chars_for_embed)
    ec.set_char2index_and_index2char(chars_for_embed)
#    print(ec.char2index)
#    print(ec.index2char)
    tweet_texts_only_embed_chars = ec.get_only_embed_chars(tweet_texts, chars_for_embed)
    print(tweet_texts_only_embed_chars)
    
    
#    context_target_onehot = ec.create_context_target_onehot_vectors(2, tweet_texts_only_embed_chars, chars_for_embed)
#    print(len(context_target_onehot[0][0]))
#    print(len(context_target_onehot))
#    print(context_target_onehot)
    

    indexed_tweet_texts = ec.get_indexed_tweet_texts(tweet_texts_only_embed_chars)
    print(indexed_tweet_texts)
    pairs = ec.get_batched_target_context_index_pairs(indexed_tweet_texts, 5, 2)
    print(pairs)
    










#    embed = nn.Embedding(4, 2)
#    input = autograd.Variable(torch.LongTensor([[0],[1]]))
#    embed(input)
#    print(embed.weight)
#    print(embed(input))
    
    
#    counter = 0
#    weights
#    sum_of_weights = 0
#    for char in chars_for_embed:
#        
#        print(math.pow(chars_for_embed[char][1], (3 / 4)))
#        sum_of_weights 
#        counter = counter + chars_for_embed[char][1]
#        



        
    
    
    
    
    
    
    
# int to onehot conversion
#    b = np.zeros((a.size, a.max()+1))
#    b[np.arange(a.size),a] = 1

    
    
    
#    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
#    raw_text = """We are about to study the idea of a computational process.
#    Computational processes are abstract beings that inhabit computers.
#    As they evolve, processes manipulate other abstract things called data.
#    The evolution of a process is directed by a pattern of rules
#    called a program. People create programs to direct processes. In effect,
#    we conjure the spirits of the computer with our spells.""".split()
#    
#    # By deriving a set from `raw_text`, we deduplicate the array
#    vocab = set(raw_text)
#    vocab_size = len(vocab)
#    
#    word_to_ix = {word: i for i, word in enumerate(vocab)}
#    data = []
#    for i in range(2, len(raw_text) - 2):
#        context = [raw_text[i - 2], raw_text[i - 1],
#                   raw_text[i + 1], raw_text[i + 2]]
#        target = raw_text[i]
#        data.append((context, target))
##    print(data[:5])
#
#
#
#
#
#    #class CBOW(nn.Module):
#    #
#    #    def __init__(self):
#    #        pass
#    #
#    #    def forward(self, inputs):
#    #        pass
#    #
#    ## create your model and train.  here are some functions to help you make
#    ## the data ready for use by your module
#    #
#    #
#    def make_context_vector(context, word_to_ix):
#        idxs = [word_to_ix[w] for w in context]
#        tensor = torch.LongTensor(idxs)
#        return autograd.Variable(tensor)
#    
#    
#    make_context_vector(data[0][0], word_to_ix)  # example
#    
##    print(make_context_vector(data[0][0], word_to_ix))


    
    
if __name__ == '__main__':
    main()
