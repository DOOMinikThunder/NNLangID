# -*- coding: utf-8 -*-

from pathlib import Path
import yaml
import torch
from input import DataSplit, InputData
from embedding import EmbeddingCalculation
from net import RNNCalculation
from evaluation import Terminal

# check if twitter module is available
try:
    from tweet_retriever import TweetRetriever
    can_use_live_tweets = True
except ImportError:
    can_use_live_tweets = False



def main():
    
    ##############
    # PARAMETERS #
    ##############
    
    # ONLY SET THIS AND THE PARAMETERS IN CORRESPONDING YAML FILE
    use_cluster_params = True                                                 # set True for use of cluster parameters
    
    
    
    # automatically determined parameters
    if(use_cluster_params):
        yaml_file = 'SystemParametersCluster.yaml'
    else:
        yaml_file = 'SystemParameters.yaml'
    with open(yaml_file, 'r') as stream:
        system_param_dict = yaml.load(stream)
        
    system_param_dict['can_use_live_tweets'] = can_use_live_tweets            # (change this if you want to sample live tweets)
    if (system_param_dict['can_use_live_tweets']):
        print('!!! TERMINAL LIVE TWEETS ON !!!')
    else:
        print('!!! TERMINAL LIVE TWEETS OFF !!!')
    system_param_dict['cuda_is_avail'] = torch.cuda.is_available()
    if (system_param_dict['cuda_is_avail']):
        print('!!! CUDA ON !!!')
    else:
        print('!!! CUDA OFF !!!')

    ############
    # TERMINAL #
    ############    
    
    # simple terminal for testing
    if (system_param_dict['run_terminal']):
        terminal = Terminal.Terminal(system_param_dict)
        terminal.run_terminal(system_param_dict['can_use_live_tweets'])
    else:

    ########################
    # DATA FILES SPLITTING #
    ########################

        # split into training, validation and test set from an original file
        files_exist = Path(system_param_dict['out_tr_data_rel_path']).is_file() and Path(system_param_dict['out_va_data_rel_path']).is_file() and Path(system_param_dict['out_te_data_rel_path']).is_file()
        if (system_param_dict['create_splitted_data_files'] or not files_exist):
            out_filenames = [system_param_dict['out_tr_data_rel_path'], system_param_dict['out_va_data_rel_path'], system_param_dict['out_te_data_rel_path']] #same size as ratios
            data_splitter = DataSplit.DataSplit()
            splitted_data = data_splitter.split_percent_of_languages(input_file=system_param_dict['input_tr_va_te_data_rel_path'],
                                                                     ratio=system_param_dict['tr_va_te_split_ratios'],
                                                                     out_filenames=out_filenames,
                                                                     shuffle_seed=system_param_dict['split_shuffle_seed'])

    ###################################
    # DATA RETRIEVAL & TRANSFORMATION #
    ###################################

        input_data = InputData.InputData()
        """
        train_set_indexed: list of every date where each character and target gets unique id, e.g. 
            ([0,1,0,1],0),([2,3,2,3],1) for ([a,b,a,b], 'de'), ([c,d,c,d], 'en)
        val_set_indexed: see train_set_indexed
        test_set_indexed: see train_set_indexed
        vocab_chars: every character occurence as a dict of character: index, occurrences, e.g. 
            {'a': (0, 700), 'b': (1, 700)}
        vocab_lang: every language occurence as a dict of language: index, occurences, e.g. 
            {'de': (0, 32)}
        """
        train_set_indexed, val_set_indexed, test_set_indexed, real_test_set_indexed, vocab_chars, vocab_lang = input_data.get_indexed_data(
            train_data_rel_path=system_param_dict['out_tr_data_rel_path'],
            validation_data_rel_path=system_param_dict['out_va_data_rel_path'],
            test_data_rel_path=system_param_dict['out_te_data_rel_path'],
            real_test_data_rel_path=system_param_dict['input_rt_data_rel_path'],
            min_char_frequency=system_param_dict['min_char_frequency'],
            fetch_only_langs=system_param_dict['fetch_only_langs'],
            fetch_only_first_x_tweets=system_param_dict['fetch_only_first_x_tweets'])
#        print(train_set_indexed, val_set_indexed, test_set_indexed)
#        print(vocab_chars)
#        print(len(vocab_chars))
#        print(vocab_lang)
#        print(len(vocab_lang))

    ######################
    # EMBEDDING TRAINING #
    ######################

        if (system_param_dict['train_embed']):
            embedding_calculation = EmbeddingCalculation.EmbeddingCalculation()
            embedding_calculation.train_embed(train_set_indexed=train_set_indexed,
                                              val_set_indexed=val_set_indexed,
                                              vocab_chars=vocab_chars,
                                              vocab_lang=vocab_lang,
                                              system_param_dict=system_param_dict)

    ################
    # RNN TRAINING #
    ################

        rnn_calculation = RNNCalculation.RNNCalculation(system_param_dict)
        
        # train RNN model
        if (system_param_dict['train_rnn']):
            rnn_calculation.train_rnn(data_sets=[train_set_indexed, val_set_indexed],
                                      vocab_chars=vocab_chars,
                                      vocab_lang=vocab_lang)

    ##############
    # EVALUATION #
    ##############

        # evaluate test set
        if (system_param_dict['eval_test_set']):
            rnn_calculation.test_rnn(data_sets=[test_set_indexed],
                                     vocab_chars=vocab_chars,
                                     vocab_lang=vocab_lang)

        # print saved model checkpoint from file
        if (system_param_dict['print_model_checkpoint_embed_weights'] != None and system_param_dict['print_rnn_model_checkpoint'] != None):
            rnn_calculation.print_model_checkpoint(vocab_chars=vocab_chars,
                                                   vocab_lang=vocab_lang,
                                                   is_rnn_model=True)
        elif (system_param_dict['print_model_checkpoint_embed_weights'] != None and system_param_dict['print_embed_model_checkpoint'] != None):
            rnn_calculation.print_model_checkpoint(vocab_chars=vocab_chars,
                                                   vocab_lang=vocab_lang,
                                                   is_rnn_model=False)
        else:
            print('Both embedding weights and model checkpoint path required!')


if __name__ == '__main__':
    main()