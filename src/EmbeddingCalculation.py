# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import unicodecsv as csv
import os
import numpy as np


class EmbeddingCalculation(object):

    def __init__(self):
        pass
        
    
    def fetch_tweet_texts_from_file(self, absolute_path_to_file):
        tweet_texts = []
        with open(absolute_path_to_file, 'rb') as file:
            reader =  csv.reader(file, delimiter=';', encoding='utf-8')
    
#            i = 0
            
            # skip first row (['\ufeff'])
            next(reader)
            for row in reader:
    
#                if(i > 100):
#                    break
    
    
#                print(i)
                tweet_texts.append(row[1])
#                print(tweet_texts)
#                i = i + 1
        
            return tweet_texts
        
        
    def get_chars_ocurring_x_times(self, tweet_texts, x):
        alrdy_ocurred_chars = {}
        chars_for_embed = {}
        count = 0
        for tweet in tweet_texts:
            for char in tweet:
                if (char in alrdy_ocurred_chars):
                    alrdy_ocurred_chars[char] = alrdy_ocurred_chars[char] + 1
                else:
                    alrdy_ocurred_chars[char] = 1
                if (alrdy_ocurred_chars[char] == x):
                    chars_for_embed[char] = count
                    count = count + 1
        return chars_for_embed
    

    def get_only_embed_chars(self, tweet_texts, chars_for_embed):
        tweet_texts_only_embed_chars = []
        for i in range(len(tweet_texts)):
            tweet_texts_only_embed_chars.append([])
            for j in range(len(tweet_texts[i])):
                if (tweet_texts[i][j] in chars_for_embed):
                    tweet_texts_only_embed_chars[i].append(tweet_texts[i][j])
        return tweet_texts_only_embed_chars
    
    
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
                    indexes.append(chars_for_embed[tweet[i - j]])
                # context chars after the target
                for j in range(1, context_window_size + 1):
                    indexes.append(chars_for_embed[tweet[i + j]])
                    
#    indexes = [chars_for_embed[tweet[i - 2]], chars_for_embed[tweet[i - 1]],
#               chars_for_embed[tweet[i + 1]], chars_for_embed[tweet[i + 2]],
                    
                # the target char
                indexes.append(chars_for_embed[tweet[i]])

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
    
    
def main():
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = cur_dir+"/data/uniformly_sampled_dl.csv"
#    abs_path = cur_dir+"/data/test.csv"
    ec = EmbeddingCalculation()
    
    tweet_texts = ec.fetch_tweet_texts_from_file(abs_path)
    chars_for_embed = ec.get_chars_ocurring_x_times(tweet_texts, 2)
#    print(chars_for_embed)
    tweet_texts_only_embed_chars = ec.get_only_embed_chars(tweet_texts, chars_for_embed)
#    print(tweet_texts_only_embed_chars)
    context_target_onehot = ec.create_context_target_onehot_vectors(2, tweet_texts_only_embed_chars, chars_for_embed)
#    print(len(context_target_onehot[0][0]))
#    print(len(context_target_onehot))
    


 

    
    
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
