# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.optim as optim
import unicodecsv as csv
import numpy as np
import math
import random
import SkipGramModel
#from tqdm import tqdm



class EmbeddingCalculation(object):

    
    def __init__(self):
        self.sampling_table = []
        
    
    def fetch_tweet_texts_from_file(self, relative_path_to_file, fetch_only_first_x_tweets=math.inf):
        tweet_texts = []
        with open(relative_path_to_file, 'rb') as file:
            reader = csv.reader(file, delimiter=';', encoding='utf-8')
            tweet_counter = 0
            # skip first row (['\ufeff'])
            next(reader)
            for row in reader:
                if (tweet_counter >= fetch_only_first_x_tweets):
                    break
                tweet_counter += 1
                
                tweet_texts.append(row[1])
            return tweet_texts
        
        
    def get_chars_occurring_min_x_times(self, tweet_texts, x):
        occurred_chars = {}
        # count occurrence of each char in the entire corpus
        for tweet in tweet_texts:
            for char in tweet:
                # if already occurred, increment counter by 1
                if (char in occurred_chars):
                    occurred_chars[char] += 1
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
                counter += 1
        return chars_for_embed
    

    def get_only_embed_chars(self, tweet_texts, chars_for_embed):
        tweet_texts_only_embed_chars = []
        for i in range(len(tweet_texts)):
            tweet_texts_only_embed_chars.append([])
            for j in range(len(tweet_texts[i])):
                if (tweet_texts[i][j] in chars_for_embed):
                    tweet_texts_only_embed_chars[i].append(tweet_texts[i][j])
        return tweet_texts_only_embed_chars
    
    
    def get_char2index_and_index2char(self, chars_for_embed):
        char2index = {}
        index2char = {}
        for char in chars_for_embed:
            char2index[char] = chars_for_embed[char][0]
            index2char[chars_for_embed[char][0]] = char
        return [char2index, index2char]
            
    
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
    
    
    def get_indexed_tweet_texts(self, tweet_texts_only_embed_chars, chars_for_embed):
        indexed_tweet_texts = []
        for i in range(len(tweet_texts_only_embed_chars)):
            indexed_tweet_texts.append([])
            for j in range(len(tweet_texts_only_embed_chars[i])):
                indexed_tweet_texts[i].append(chars_for_embed[tweet_texts_only_embed_chars[i][j]][0])
        return indexed_tweet_texts
    
    
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
    
    
    def set_sampling_table(self, chars_for_embed, table_size=100000000):
        char_pow_frequencies = {}
        char_pow_frequencies_acc = 0
        for char in chars_for_embed:
            char_pow_frequency = math.pow(chars_for_embed[char][1], 0.75)
            char_pow_frequencies_acc = char_pow_frequencies_acc + char_pow_frequency
            char_pow_frequencies[chars_for_embed[char][0]] = char_pow_frequency
        
        # get the number of occurrences of each char in the table (depending on the probability function)
        # and fill the table accordingly
        for char_index in char_pow_frequencies:
            num_of_char = np.round((char_pow_frequencies[char_index] / char_pow_frequencies_acc) * table_size)

            for i in range(int(num_of_char)):
                self.sampling_table.append(char_index)
                
                
    def get_neg_samples(self, num_pairs, num_samples):
        return np.random.choice(self.sampling_table, size=(num_pairs, num_samples)).tolist()
            
            
    
def main():
    
    ##############
    # PARAMETERS #
    ##############
    
#    rel_path = "../../data/uniformly_sampled_dl.csv"
    rel_path = "../../data/test.csv"
    
    # Hyperparameters
    min_char_frequency = 2
    sampling_table_size = 1000
    batch_size = 10
    max_context_window_size = 2
    num_neg_samples = 5
    embed_dim = 2
    initial_lr = 0.025
    num_epochs = 1
    
    
    ###################################
    # DATA RETRIEVAL & TRANSFORMATION #
    ###################################
    
    embedding_calculation = EmbeddingCalculation()
    tweet_texts = embedding_calculation.fetch_tweet_texts_from_file(rel_path, fetch_only_first_x_tweets=math.inf)
#    print(tweet_texts)
    chars_for_embed = embedding_calculation.get_chars_occurring_min_x_times(tweet_texts, min_char_frequency)
#    print(chars_for_embed)
    tweet_texts_only_embed_chars = embedding_calculation.get_only_embed_chars(tweet_texts, chars_for_embed)
#    print(tweet_texts_only_embed_chars)
    indexed_tweet_texts = embedding_calculation.get_indexed_tweet_texts(tweet_texts_only_embed_chars, chars_for_embed)
#    print(indexed_tweet_texts)
    
    
    ##########################################
    # SKIP-GRAM-MODEL WITH NEGATIVE SAMPLING #
    ##########################################
    
    embedding_calculation.set_sampling_table(chars_for_embed, sampling_table_size)
#    print(embedding_calculation.sampling_table)
#    print(len(embedding_calculation.sampling_table))
    batched_pairs = embedding_calculation.get_batched_target_context_index_pairs(indexed_tweet_texts, batch_size, max_context_window_size)
#    print(batched_pairs)
    skip_gram_model = SkipGramModel.SkipGramModel(len(chars_for_embed), embed_dim)
    optimizer = optim.SGD(skip_gram_model.parameters(), lr=initial_lr)
    
    # train skip-gram
    num_epochs_minus_one = num_epochs - 1
    for epoch_i in range(num_epochs):
        print("Epoch:", epoch_i, "/", num_epochs_minus_one)
        for batch in batched_pairs:
            targets = [pair[0] for pair in batch]
            contexts = [pair[1] for pair in batch]
            neg_samples = embedding_calculation.get_neg_samples(len(batch), num_neg_samples)
#            print(neg_samples)
            
            optimizer.zero_grad()
            loss = skip_gram_model.forward(targets, contexts, neg_samples)
            loss.backward()
            optimizer.step()
    
    
    
    
    
    
    ###########
    # TESTING #
    ###########
    
    print("VOCABULARY:", chars_for_embed)
    print("VOCABULARY SIZE:", len(chars_for_embed))
    
    a = skip_gram_model.embed_hidden(autograd.Variable(torch.LongTensor([[0]])))
    b = skip_gram_model.embed_hidden(autograd.Variable(torch.LongTensor([[1]])))
    c = skip_gram_model.embed_hidden(autograd.Variable(torch.LongTensor([[2]])))
    d = skip_gram_model.embed_hidden(autograd.Variable(torch.LongTensor([[3]])))
    
    print("EMBEDDING VECTOR DIFFERENCES:")
    print("a-b", torch.FloatTensor.sum((torch.abs(a - b)).data[0]))
    print("c-d", torch.FloatTensor.sum((torch.abs(c - d)).data[0]))
    print("a-c", torch.FloatTensor.sum((torch.abs(a - c)).data[0]))
    print("a-d", torch.FloatTensor.sum((torch.abs(a - d)).data[0]))
    print("b-c", torch.FloatTensor.sum((torch.abs(b - c)).data[0]))
    print("b-d", torch.FloatTensor.sum((torch.abs(b - d)).data[0]))
    
    
    
    
    
    
    
#    context_target_onehot = embedding_calculation.create_context_target_onehot_vectors(2, tweet_texts_only_embed_chars, chars_for_embed)
#    print(len(context_target_onehot[0][0]))
#    print(len(context_target_onehot))
#    print(context_target_onehot)
    
    
# int to onehot conversion
#    b = np.zeros((a.size, a.max()+1))
#    b[np.arange(a.size),a] = 1


    
if __name__ == '__main__':
    main()
