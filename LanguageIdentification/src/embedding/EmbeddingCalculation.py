# -*- coding: utf-8 -*-

#    MIT License
#    
#    Copyright (c) 2018 Alexander Heilig, Dominik Sauter, Tabea Kiupel
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.


import math
import torch
from torch.autograd import Variable
from input import InputData
from embedding import SkipGramModel
#from tqdm import tqdm


class EmbeddingCalculation(object):
    """Class containing the set-up and execution of the embedding training,
    with some basic testing functionality.
    """

    def train_embed(self, train_set_indexed, val_set_indexed, vocab_chars, vocab_lang, system_param_dict):
        """
        Create embedding model and batched pairs for training, then train the model.
        After training, a basic test may be printed.
        
        Args:
            train_set_indexed: The indexed training set (list of tuples).
            val_set_indexed: The indexed validation set (list of tuples).
            vocab_chars: Every character occurence as a dict of {character: (index, occurrences)}.
            vocab_lang: Every language occurence as a dict of {language: (index, occurences)}.
            system_param_dict: Dict containing the system parameters.
        """
        # set embedding dimension to: roundup(log2(vocabulary-size))
        embed_dim = math.ceil(math.log(len(vocab_chars),2))
        
        ##########################################
        # SKIP-GRAM-MODEL WITH NEGATIVE SAMPLING #
        ##########################################
        
        input_data = InputData.InputData()
        train_indexed_texts = input_data.get_only_indexed_texts(train_set_indexed)
        val_indexed_texts = input_data.get_only_indexed_texts(val_set_indexed)
        
        batch_size = system_param_dict['batch_size_rnn']
        max_context_window_size = system_param_dict['max_context_window_size']
        train_batched_pairs = input_data.get_batched_target_context_index_pairs(train_indexed_texts, batch_size, max_context_window_size)
        val_batched_pairs = input_data.get_batched_target_context_index_pairs(val_indexed_texts, batch_size, max_context_window_size)
        
        skip_gram_model = SkipGramModel.SkipGramModel(vocab_chars=vocab_chars,
                                                      vocab_lang=vocab_lang,
                                                      embed_dim=embed_dim,
                                                      system_param_dict=system_param_dict)
        if (system_param_dict['cuda_is_avail']):
            skip_gram_model.cuda()
        
        # train skip-gram with negative sampling
        skip_gram_model.train(train_batched_pairs=train_batched_pairs,
                              val_batched_pairs=val_batched_pairs)
                             
        ###########
        # TESTING #
        ###########
        
        if (system_param_dict['print_embed_testing']):
            input_data = InputData.InputData()
            embed, num_classes = input_data.create_embed_from_weights_file(system_param_dict['embed_weights_rel_path'])
            char2index, index2char = input_data.get_string2index_and_index2string(vocab_chars)
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