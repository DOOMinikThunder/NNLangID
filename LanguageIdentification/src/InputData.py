# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import unicodecsv as csv
import math



class InputData(object):
        
        
    def fetch_tweet_texts_and_lang_from_file(self, relative_path_to_file, fetch_only_langs=None, fetch_only_first_x_tweets=math.inf):
        texts_and_lang = []
        with open(relative_path_to_file, 'rb') as file:
            reader = csv.reader(file, delimiter=';', encoding='utf-8')
            tweet_counter = 0
            # skip first row (['\ufeff'])
            next(reader)
            for row in reader:
                if (tweet_counter >= fetch_only_first_x_tweets):
                    break
                tweet_counter += 1
                
                # if only tweets of specific languages shall be fetched
                if (fetch_only_langs != None):
                    for lang in fetch_only_langs:
                        if (row[2] == lang):
                            texts_and_lang.append((row[1], row[2]))
                else:
                    texts_and_lang.append((row[1], row[2]))
        return texts_and_lang
        
        
    def get_vocab_chars_and_lang(self, texts_and_lang, min_char_frequency):
        occurred_chars = {}
        occurred_langs = {}
        # count occurrence of each char and language in the entire corpus
        for tweet in texts_and_lang:
            # chars:
            for char in tweet[0]:
                # if already occurred, increment counter by 1
                if (char in occurred_chars):
                    occurred_chars[char] += 1
                # if occurred for the first time, add to dict with count 1
                else:
                    occurred_chars[char] = 1
            # language:
            # if already occurred, increment counter by 1
            if (tweet[1] in occurred_langs):
                occurred_langs[tweet[1]] += 1
            # if occurred for the first time, add to dict with count 1
            else:
                occurred_langs[tweet[1]] = 1
            
        # fill new dict with the chars that occurred at least min_char_frequency times
        # (filled values are tuples: (onehot-index, number of occurrences))
        vocab_chars = {}
        char_counter = 0
        for char in occurred_chars:
            if (occurred_chars[char] >= min_char_frequency):
                vocab_chars[char] = (char_counter, occurred_chars[char])
                char_counter += 1
        
        # fill new dict with the occurred languages and their frequency
        # (filled values are tuples: (onehot-index, number of occurrences))
        vocab_lang = {}
        lang_counter = 0
        for lang in occurred_langs:
            vocab_lang[lang] = (lang_counter, occurred_langs[lang])
            lang_counter += 1
                
        return vocab_chars, vocab_lang
    

    def get_texts_with_only_vocab_chars(self, texts_and_lang, vocab_chars):
        texts_and_lang_only_vocab_chars = []
        for i in range(len(texts_and_lang)):
            temp_text = []
            for j in range(len(texts_and_lang[i][0])):
                if (texts_and_lang[i][0][j] in vocab_chars):
                    temp_text.append(texts_and_lang[i][0][j])
            texts_and_lang_only_vocab_chars.append((temp_text, texts_and_lang[i][1]))
        return texts_and_lang_only_vocab_chars
    
    
    def get_indexed_texts_and_lang(self, texts_and_lang_only_vocab_chars, vocab_chars, vocab_lang):
        indexed_texts_and_lang = []
        for i in range(len(texts_and_lang_only_vocab_chars)):
            temp_indexed_text = []
            for j in range(len(texts_and_lang_only_vocab_chars[i][0])):
                temp_indexed_text.append(vocab_chars[texts_and_lang_only_vocab_chars[i][0][j]][0])
            indexed_texts_and_lang.append((temp_indexed_text, vocab_lang[texts_and_lang_only_vocab_chars[i][1]][0]))
        return indexed_texts_and_lang
      
    
    def get_indexed_data(self, input_data_rel_path, min_char_frequency, fetch_only_langs=None, fetch_only_first_x_tweets=math.inf):
        texts_and_lang = self.fetch_tweet_texts_and_lang_from_file(input_data_rel_path, fetch_only_langs, fetch_only_first_x_tweets)
#        print(texts_and_lang)
        vocab_chars, vocab_lang = self.get_vocab_chars_and_lang(texts_and_lang, min_char_frequency)
#        print(vocab_chars)
#        print(vocab_lang)
        texts_and_lang_only_vocab_chars = self.get_texts_with_only_vocab_chars(texts_and_lang, vocab_chars)
#        print(texts_and_lang_only_vocab_chars)
        indexed_texts_and_lang = self.get_indexed_texts_and_lang(texts_and_lang_only_vocab_chars, vocab_chars, vocab_lang)
#        print(indexed_texts_and_lang)
        return indexed_texts_and_lang, vocab_chars, vocab_lang
    
    
    def get_char2index_and_index2char(self, vocab_chars):
        char2index = {}
        index2char = {}
        for char in vocab_chars:
            char2index[char] = vocab_chars[char][0]
            index2char[vocab_chars[char][0]] = char
        return char2index, index2char
    
    
    def get_only_indexed_texts(self, indexed_texts_and_lang):
        indexed_texts = []
        for i in range(len(indexed_texts_and_lang)):
            indexed_texts.append(indexed_texts_and_lang[i][0])
        return indexed_texts
    
    
    def create_embed_from_weights_file(self, relative_path_to_file):
        weights = []
        embed_dims = []
        with open(relative_path_to_file, 'rb') as file:
            reader = csv.reader(file, delimiter=' ', encoding='utf-8')

            first_row = True
            for row in reader:
                if (first_row):
                    embed_dims = [int(x) for x in row]
                    first_row = False
                else:
                    float_row = [float(x) for x in row]
                    weights.append(float_row)

            weights_tensor = torch.FloatTensor(weights)
            weights_tensor_param = torch.nn.Parameter(weights_tensor, requires_grad=False)
            embed = torch.nn.Embedding(embed_dims[0], embed_dims[1])
            embed.weight = weights_tensor_param
        return embed
    
    
    def create_embed_input_and_target_tensors(self, indexed_texts_and_lang, embed_weights_rel_path):
        embed = self.create_embed_from_weights_file(embed_weights_rel_path)

        embed_char_text_inp_tensors = []
        target_tensors = []
        for tweet in indexed_texts_and_lang:
            # tweet text:
            # get tensor with the embedding for each char of the tweet
            embed_tensor = embed(Variable(torch.LongTensor(tweet[0])))
            # create correctly dimensionated input tensor and append to input tensor list
            dims = list(embed_tensor.size())
            embed_tensor_inp = embed_tensor.view(dims[0], -1, dims[1])
            embed_char_text_inp_tensors.append(embed_tensor_inp)
            # tweet language:
            # create list in size of the number of chars of the tweet and fill with language index
            target_list = [tweet[1] for x in range(dims[0])]
            # create target tensor and append to target tensor list
            target_list_tensor = Variable(torch.LongTensor(target_list))
            target_tensors.append(target_list_tensor)
        return embed_char_text_inp_tensors, target_tensors