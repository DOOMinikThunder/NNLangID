# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import unicodecsv as csv
import math
import random



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
    
    
    def filter_out_irrelevant_tweet_parts(self, texts_and_lang):
        filtered_texts_and_lang = []
        removal_mode = False
        for tweet_i in range(len(texts_and_lang)):
            filtered_tweet_text = []
            tweet_text_size = len(texts_and_lang[tweet_i][0])
            for char_j in range(tweet_text_size):
                # check for hashtags and @-names and activate removal mode (keep '#' or '@')
                if (not removal_mode
                    and texts_and_lang[tweet_i][0][char_j] == '#'
                    or texts_and_lang[tweet_i][0][char_j] == '@'):
                    removal_mode = True
                    filtered_tweet_text.append(texts_and_lang[tweet_i][0][char_j])
                # check for URLs and activate removal mode (replace with '_')
                elif (not removal_mode
                      and texts_and_lang[tweet_i][0][char_j] == 'h'):
                    # check if not out of bounds and 'http://'
                    if (char_j+6 < tweet_text_size
                        and texts_and_lang[tweet_i][0][char_j+1] == 't'
                        and texts_and_lang[tweet_i][0][char_j+2] == 't'
                        and texts_and_lang[tweet_i][0][char_j+3] == 'p'
                        and texts_and_lang[tweet_i][0][char_j+4] == ':'
                        and texts_and_lang[tweet_i][0][char_j+5] == '/'
                        and texts_and_lang[tweet_i][0][char_j+6] == '/'):
                        removal_mode = True
                        filtered_tweet_text.append('_')
                    # check if not out of bounds and 'https://'
                    elif (char_j+7 < tweet_text_size
                             and texts_and_lang[tweet_i][0][char_j+1] == 't'
                             and texts_and_lang[tweet_i][0][char_j+2] == 't'
                             and texts_and_lang[tweet_i][0][char_j+3] == 'p'
                             and texts_and_lang[tweet_i][0][char_j+4] == 's'
                             and texts_and_lang[tweet_i][0][char_j+5] == ':'
                             and texts_and_lang[tweet_i][0][char_j+6] == '/'
                             and texts_and_lang[tweet_i][0][char_j+7] == '/'):
                        removal_mode = True
                        filtered_tweet_text.append('_')
                    # append char as it is a normal 'h' ocurrence
                    else:
                        filtered_tweet_text.append(texts_and_lang[tweet_i][0][char_j])
                # check if part to be removed has ended to quit removal mode (append ' ')
                elif (removal_mode and texts_and_lang[tweet_i][0][char_j] == ' '):
                    removal_mode = False
                    filtered_tweet_text.append(texts_and_lang[tweet_i][0][char_j])
                # append char if removal mode is not active
                elif (not removal_mode):
                    filtered_tweet_text.append(texts_and_lang[tweet_i][0][char_j])
            # if there is still text: append tweet to list
            if (filtered_tweet_text != []):
                filtered_texts_and_lang.append((''.join(filtered_tweet_text), texts_and_lang[tweet_i][1]))
        return filtered_texts_and_lang
    
    
    # !!! DEPRECATED !!!
    # set_ratios must be: [train_ratio, val_ratio, test_ratio]
    def split_data_into_sets(self, texts_and_lang, set_ratios):
        if (set_ratios[0] + set_ratios[1] != 1):
            print("Error: Set ratios do not sum to 1!")
            return -1
        data_size = len(texts_and_lang)
        val_size = int(set_ratios[1] * data_size)
        # train set size is adapted to fit total data size
        # (as it is usually the largest set, the error will be neglectible)
        train_size = data_size - (val_size)

        train_set = []
        val_set = []
        test_set = []
        for i in range(train_size):
            train_set.append(texts_and_lang[i])
        for i in range(val_size):
            val_set.append(texts_and_lang[i+train_size])
        return train_set, val_set
    
        
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
            if(temp_text != []):
                texts_and_lang_only_vocab_chars.append((temp_text, texts_and_lang[i][1]))
        return texts_and_lang_only_vocab_chars
    
    
    def get_indexed_texts_and_lang(self, texts_and_lang_only_vocab_chars, vocab_chars, vocab_lang):
        indexed_texts_and_lang = []
        for i in range(len(texts_and_lang_only_vocab_chars)):
            temp_indexed_text = []
            for j in range(len(texts_and_lang_only_vocab_chars[i][0])):
                if texts_and_lang_only_vocab_chars[i][0][j] in vocab_chars:
                    temp_indexed_text.append(vocab_chars[texts_and_lang_only_vocab_chars[i][0][j]][0])
            if texts_and_lang_only_vocab_chars[i][1] in vocab_lang:
                indexed_texts_and_lang.append((temp_indexed_text, vocab_lang[texts_and_lang_only_vocab_chars[i][1]][0]))
        return indexed_texts_and_lang
    
    
    def get_indexed_data(self, train_data_rel_path, validation_data_rel_path, test_data_rel_path, real_test_data_rel_path, min_char_frequency, fetch_only_langs=None, fetch_only_first_x_tweets=math.inf):
        tr_texts_and_lang = self.fetch_tweet_texts_and_lang_from_file(train_data_rel_path, fetch_only_langs, fetch_only_first_x_tweets)
        va_texts_and_lang = self.fetch_tweet_texts_and_lang_from_file(validation_data_rel_path, fetch_only_langs, fetch_only_first_x_tweets)
        te_texts_and_lang = self.fetch_tweet_texts_and_lang_from_file(test_data_rel_path, fetch_only_langs, fetch_only_first_x_tweets)
        rt_texts_and_lang = self.fetch_tweet_texts_and_lang_from_file(real_test_data_rel_path, fetch_only_langs, fetch_only_first_x_tweets)
        
        # (true) randomly shuffle each data set
        random.shuffle(tr_texts_and_lang)
        random.shuffle(va_texts_and_lang)
        random.shuffle(te_texts_and_lang)
        random.shuffle(rt_texts_and_lang)
        
        filtered_tr_texts_and_lang = self.filter_out_irrelevant_tweet_parts(tr_texts_and_lang)
        filtered_va_texts_and_lang = self.filter_out_irrelevant_tweet_parts(va_texts_and_lang)
        filtered_te_texts_and_lang = self.filter_out_irrelevant_tweet_parts(te_texts_and_lang)
        filtered_rt_texts_and_lang = self.filter_out_irrelevant_tweet_parts(rt_texts_and_lang)
        
        vocab_chars, vocab_lang = self.get_vocab_chars_and_lang(filtered_tr_texts_and_lang, min_char_frequency)

        train_set_only_vocab_chars = self.get_texts_with_only_vocab_chars(filtered_tr_texts_and_lang, vocab_chars)
        val_set_only_vocab_chars = self.get_texts_with_only_vocab_chars(filtered_va_texts_and_lang, vocab_chars)
        test_set_only_vocab_chars = self.get_texts_with_only_vocab_chars(filtered_te_texts_and_lang, vocab_chars)
        real_test_set_only_vocab_chars = self.get_texts_with_only_vocab_chars(filtered_rt_texts_and_lang, vocab_chars)

        train_set_indexed = self.get_indexed_texts_and_lang(train_set_only_vocab_chars, vocab_chars, vocab_lang)
        val_set_indexed = self.get_indexed_texts_and_lang(val_set_only_vocab_chars, vocab_chars, vocab_lang)
        test_set_indexed = self.get_indexed_texts_and_lang(test_set_only_vocab_chars, vocab_chars, vocab_lang)
        real_test_set_indexed = self.get_indexed_texts_and_lang(real_test_set_only_vocab_chars, vocab_chars, vocab_lang)

        return train_set_indexed, val_set_indexed, test_set_indexed, real_test_set_indexed, vocab_chars, vocab_lang
    
    
    def get_string2index_and_index2string(self, vocab_dict):
        string2index = {}
        index2string = {}
        for string in vocab_dict:
            string2index[string] = vocab_dict[string][0]
            index2string[vocab_dict[string][0]] = string
        return string2index, index2string
    
    
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
            num_classes = embed_dims[2]     # embed_dims[2] is the number of classes
        print('Embedding weights loaded from file:', relative_path_to_file)
        return embed, num_classes
    
    
    def create_embed_input_and_target_tensors(self, indexed_texts_and_lang, embed_weights_rel_path):
        embed, num_classes = self.create_embed_from_weights_file(embed_weights_rel_path)

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