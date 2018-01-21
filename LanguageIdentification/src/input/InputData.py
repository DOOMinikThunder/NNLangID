# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import unicodecsv as csv
import math
import random



class InputData(object):
        
        
    def __fetch_tweet_texts_and_lang_from_file(self, relative_path_to_file, fetch_only_langs=None, fetch_only_first_x_tweets=float('inf')):
        """

        Args:
            relative_path_to_file:
            fetch_only_langs:
            fetch_only_first_x_tweets:

        Returns:

        """
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
        """

        Args:
            texts_and_lang:

        Returns:

        """
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
        """

        Args:
            texts_and_lang:
            set_ratios:

        Returns:

        """
        if (set_ratios[0] + set_ratios[1] != 1):
            print("Error: Set ratios do not sum to 1!")
            return -1
        data_size = len(texts_and_lang)
        val_size = int(set_ratios[1] * data_size)
        # train set size is adapted to fit total tweet_retriever_data size
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
    
        
    def __get_vocab_chars_and_lang(self, texts_and_lang, min_char_frequency):
        """

        Args:
            texts_and_lang:
            min_char_frequency:

        Returns:

        """
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
        """

        Args:
            texts_and_lang:
            vocab_chars:

        Returns:

        """
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
        """

        Args:
            texts_and_lang_only_vocab_chars:
            vocab_chars:
            vocab_lang:

        Returns:

        """
        indexed_texts_and_lang = []
        for i in range(len(texts_and_lang_only_vocab_chars)):
            temp_indexed_text = []
            for j in range(len(texts_and_lang_only_vocab_chars[i][0])):
                if texts_and_lang_only_vocab_chars[i][0][j] in vocab_chars:
                    temp_indexed_text.append(vocab_chars[texts_and_lang_only_vocab_chars[i][0][j]][0])
            if texts_and_lang_only_vocab_chars[i][1] in vocab_lang:
                indexed_texts_and_lang.append((temp_indexed_text, vocab_lang[texts_and_lang_only_vocab_chars[i][1]][0]))
        return indexed_texts_and_lang
    
    
    def get_indexed_data(self, train_data_rel_path, validation_data_rel_path, test_data_rel_path, real_test_data_rel_path, min_char_frequency, fetch_only_langs=None, fetch_only_first_x_tweets=float('inf')):
        """Gets all relevant data in indexed form, as well as the vocabularies, to be readily used by the embedding and RNN.

        Args:
            train_data_rel_path: Relative path to training set file.
            validation_data_rel_path: Relative path to validation set file.
            test_data_rel_path: Relative path to test set file.
            real_test_data_rel_path: Relative path to real test set file.
            min_char_frequency: Minimum character frequency for a character in the training set to be in the vocabulary (and later used).
            fetch_only_langs: Fetch only the as a list of language tags specified languages. If 'None', all languages will be fetched.
            fetch_only_first_x_tweets: Fetch only the first x amount of tweets in the files. Set to infinity to fetch all tweets.

        Returns:
            train_set_indexed: List of tuples, with each tuple representing one tweet in the data set with the data set being (true) randomly shuffled each time.
                The first tuple element is the preprocessed tweet text as a list of characters, the second is the language.
                Thereby, both characters and languages are replaced by their unique indices,
                which are obtained by the two vocabularies for characters (vocab_chars) and languages (vocab_lang).
                Example for a set with two tweets:
                    [([0,1,0,1], 0), ([2,3,2,3], 1)] for [([a,b,a,b], 'de'), ([c,d,c,d], 'en')]
            val_set_indexed: See train_set_indexed.
            test_set_indexed: See train_set_indexed.
            real_test_set_indexed: See train_set_indexed.
            vocab_chars: Dict for a mapping of each ocurred character in the training data with frequency >= min_char_frequency
                to a tuple of unique index and frequency of its occurence. In the form: {character: (index, frequency)}.
                Example for two occured characters:
                    {'a': (0, 1337), 'b': (1, 42)}
            vocab_lang: The same as vocab_chars, but for languages instead of characters (and no frequency threshold).
        """
        tr_filtered = self.__get_filtered_data(train_data_rel_path, min_char_frequency, fetch_only_langs, fetch_only_first_x_tweets, calc_vocab=True)
        vocab_chars, vocab_lang = self.__get_vocab_chars_and_lang(tr_filtered, min_char_frequency)
        val_filtered =  self.__get_filtered_data(validation_data_rel_path, min_char_frequency, fetch_only_langs, fetch_only_first_x_tweets)
        te_filtered =  self.__get_filtered_data(test_data_rel_path, min_char_frequency, fetch_only_langs, fetch_only_first_x_tweets)
        rt_filtered =  self.__get_filtered_data(real_test_data_rel_path, min_char_frequency, fetch_only_langs, fetch_only_first_x_tweets)
        tr_indexed = self.__get_single_indexed_data(tr_filtered, vocab_lang, vocab_chars)
        val_indexed = self.__get_single_indexed_data(val_filtered, vocab_lang, vocab_chars)
        te_indexed = self.__get_single_indexed_data(te_filtered, vocab_lang, vocab_chars)
        rt_indexed = self.__get_single_indexed_data(rt_filtered, vocab_lang, vocab_chars)
        return tr_indexed, val_indexed, te_indexed, rt_indexed, vocab_chars, vocab_lang


    def __get_filtered_data(self, data_path, min_char_frequency, fetch_only_langs=None, fetch_only_first_x_tweets=float('inf'), calc_vocab=False):
        """

        Args:
            data_path:
            min_char_frequency:
            fetch_only_langs:
            fetch_only_first_x_tweets:
            calc_vocab:

        Returns:

        """
        texts_and_lang = self.__fetch_tweet_texts_and_lang_from_file(data_path, fetch_only_langs, fetch_only_first_x_tweets)
        random.shuffle(texts_and_lang)
        filtered_texts_and_lang = self.filter_out_irrelevant_tweet_parts(texts_and_lang)
        return filtered_texts_and_lang


    def __get_single_indexed_data(self, filtered_texts_and_lang, vocab_lang, vocab_chars):
        """

        Args:
            filtered_texts_and_lang:
            vocab_lang:
            vocab_chars:

        Returns:

        """
        set_only_vocab_chars = self.get_texts_with_only_vocab_chars(filtered_texts_and_lang, vocab_chars)
        set_indexed = self.get_indexed_texts_and_lang(set_only_vocab_chars, vocab_chars, vocab_lang)
        return set_indexed


    def get_string2index_and_index2string(self, vocab_dict):
        """

        Args:
            vocab_dict:

        Returns:

        """
        string2index = {}
        index2string = {}
        for string in vocab_dict:
            string2index[string] = vocab_dict[string][0]
            index2string[vocab_dict[string][0]] = string
        return string2index, index2string
    
    
    def get_only_indexed_texts(self, indexed_texts_and_lang):
        """

        Args:
            indexed_texts_and_lang:

        Returns:

        """
        indexed_texts = []
        for i in range(len(indexed_texts_and_lang)):
            indexed_texts.append(indexed_texts_and_lang[i][0])
        return indexed_texts
    
    
    def get_batched_target_context_index_pairs(self, indexed_tweet_texts, batch_size, max_window_size):
        """

        Args:
            indexed_tweet_texts:
            batch_size:
            max_window_size:

        Returns:

        """
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
    
    
    def create_embed_input_and_target_tensors(self, indexed_texts_and_lang, embed_weights_rel_path, embed=None):
        # if no embed is passed: get it first
        if (embed == None):
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
    
    
    # !!! UNUSED !!!
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
    
    
    # !!! UNUSED !!!
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