# -*- coding: utf-8 -*-

import math
from random import shuffle
import InputData
from embedding import EmbeddingCalculation
from net import GRUModel



def main():
    
    ##############
    # PARAMETERS #
    ##############
    
#    input_data_rel_path = "../data/input_data/uniformly_sampled_dl.csv"
    input_data_rel_path = "../data/input_data/test.csv"
    embed_weights_rel_path = "../data/embed_weights/embed_weights.txt"
    fetch_only_langs = None#['el', 'fa', 'hi', 'ca']
    fetch_only_first_x_tweets = math.inf#5
    calc_embed = True
    
    # HYPERPARAMETERS EMBEDDING
    min_char_frequency = 2
    sampling_table_size = 1000
    batch_size_embed = 2
    max_context_window_size = 2
    num_neg_samples = 5
#    embed_dim = 2   # will be set automatically later
    initial_lr_embed = 0.025
    num_epochs_embed = 1
    
    # HYPERPARAMETERS RNN
#    input_size = list(embed_char_text_inp_tensors[0].size())[2]
#    num_classes = len(vocab_lang)
    hidden_size = 100
    num_layers = 1
    batch_size_rnn = 1
    num_epochs_rnn = 1
    num_batches_rnn = 1
    is_bidirectional = True
    
    
    ###################################
    # DATA RETRIEVAL & TRANSFORMATION #
    ###################################
    
    input_data = InputData.InputData()
    """
    indexed_texts_and_lang: list of every date where each character and target gets unique id, e.g. 
        ([0,1,0,1],0),([2,3,2,3],1) for ([a,b,a,b], 'de'), ([c,d,c,d], 'en)
    vocab_chars: every character occurence as a dict of character: index, occurrences, e.g. 
        {'a': (0, 700), 'b': (1, 700)}
    vocab_lang: every language occurence as a dict of language: index, occurences, e.g. 
        {'de': (0, 32)}
    """
    indexed_texts_and_lang, vocab_chars, vocab_lang = input_data.get_indexed_data(input_data_rel_path=input_data_rel_path,
                                                                                  min_char_frequency=min_char_frequency,
                                                                                  fetch_only_langs=fetch_only_langs,
                                                                                  fetch_only_first_x_tweets=fetch_only_first_x_tweets)
#    print(indexed_texts_and_lang)
#    print(vocab_chars)
#    print(vocab_lang)
    

    #########################
    # EMBEDDING CALCULATION #
    #########################
    
    if (calc_embed):
        indexed_texts = input_data.get_only_indexed_texts(indexed_texts_and_lang)
        #print('indexed texts', indexed_texts)
        embedding_calculation = EmbeddingCalculation.EmbeddingCalculation()
        embedding_calculation.calc_embed(indexed_tweet_texts=indexed_texts,
                                         batch_size=batch_size_embed,
                                         vocab_chars=vocab_chars,
                                         max_context_window_size=max_context_window_size,
                                         num_neg_samples=num_neg_samples,
                                         sampling_table_size=sampling_table_size,
                                         num_epochs=num_epochs_embed,
                                         initial_lr=initial_lr_embed,
                                         embed_weights_rel_path=embed_weights_rel_path)
    
    
    ################
    # RNN TRAINING #
    ################
 
    # initialization
    embed_char_text_inp_tensors, target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=indexed_texts_and_lang,
                                                                                                   embed_weights_rel_path=embed_weights_rel_path)
    #print('input size', list(embed_char_text_inp_tensors[0].size())[2])
    gru_model = GRUModel.GRUModel(input_size=list(embed_char_text_inp_tensors[0].size())[2],
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  num_classes=len(vocab_lang),
                                  is_bidirectional=is_bidirectional)
    print('MODEL:\n', gru_model)

    # training
    # input: whole data set, every date contains the embedding of one char in one dimension
    # target: whole target set, target is set for each character embedding
    idx_list = [i for i in range(len(embed_char_text_inp_tensors))]
    shuffle(idx_list)
    shuffled_input = [embed_char_text_inp_tensors[i] for i in idx_list]
    shuffled_target = [target_tensors[i] for i in idx_list]

    print('input', len(embed_char_text_inp_tensors))
    print('target', len(target_tensors))
    gru_model.train(inputs=shuffled_input,
                    targets=shuffled_target,
                    batch_size=batch_size_rnn,
                    num_batches=num_batches_rnn,
                    num_epochs=num_epochs_rnn)

    #evaluate validation set
    gru_model.train(inputs=embed_char_text_inp_tensors,
                    targets=target_tensors,
                    batch_size=batch_size_rnn,
                    num_batches=num_batches_rnn,
                    num_epochs=1,
                    eval=True)



#    # training
#    gru_model.train(inputs=embed_char_text_inp_tensors,
#                    targets=target_tensors,
#                    batch_size=batch_size_rnn,
#                    num_batches=num_batches_rnn,
#                    num_epochs=num_epochs_rnn)
    

    
    
    
    

if __name__ == '__main__':
    main()