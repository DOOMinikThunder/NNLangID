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
    use_cluster_params = False                                                 # set True for use of cluster parameters
    
    
    
    # automatically determined parameters
    if(use_cluster_params):
        yaml_file = 'SystemParametersCluster.yaml'
    else:
        yaml_file = 'SystemParameters.yaml'
    with open(yaml_file, 'r') as stream:
        system_parameters = yaml.load(stream)
        
    system_parameters['can_use_live_tweets'] = can_use_live_tweets            # (change this if you want to sample live tweets)
    if (system_parameters['can_use_live_tweets']):
        print('!!! TERMINAL LIVE TWEETS ON !!!')
    else:
        print('!!! TERMINAL LIVE TWEETS OFF !!!')
    system_parameters['cuda_is_avail'] = torch.cuda.is_available()
    if (system_parameters['cuda_is_avail']):
        print('!!! CUDA ON !!!')
    else:
        print('!!! CUDA OFF !!!')

    ############
    # TERMINAL #
    ############    
    
    # simple terminal for testing
    if (system_parameters['run_terminal']):
        terminal = Terminal.Terminal(system_parameters)
        terminal.run_terminal(system_parameters['can_use_live_tweets'])
    else:

    ########################
    # DATA FILES SPLITTING #
    ########################

        # split into training, validation and test set from an original file
        files_exist = Path(system_parameters['out_tr_data_rel_path']).is_file() and Path(system_parameters['out_va_data_rel_path']).is_file() and Path(system_parameters['out_te_data_rel_path']).is_file()
        if (system_parameters['create_splitted_data_files'] or not files_exist):
            out_filenames = [system_parameters['out_tr_data_rel_path'], system_parameters['out_va_data_rel_path'], system_parameters['out_te_data_rel_path']] #same size as ratios
            data_splitter = DataSplit.DataSplit()
            splitted_data = data_splitter.split_percent_of_languages(system_parameters['input_tr_va_te_data_rel_path'],
                                                                     system_parameters['tr_va_te_split_ratios'],
                                                                     out_filenames,
                                                                     system_parameters['split_shuffle_seed'])

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
            train_data_rel_path=system_parameters['out_tr_data_rel_path'],
            validation_data_rel_path=system_parameters['out_va_data_rel_path'],
            test_data_rel_path=system_parameters['out_te_data_rel_path'],
            real_test_data_rel_path=system_parameters['input_rt_data_rel_path'],
            min_char_frequency=system_parameters['min_char_frequency'],
            fetch_only_langs=system_parameters['fetch_only_langs'],
            fetch_only_first_x_tweets=system_parameters['fetch_only_first_x_tweets'])
#        print(train_set_indexed, val_set_indexed, test_set_indexed)
#        print(vocab_chars)
#        print(len(vocab_chars))
#        print(vocab_lang)
#        print(len(vocab_lang))

    #########################
    # EMBEDDING CALCULATION #
    #########################

        if (system_parameters['calc_embed']):
            embedding_calculation = EmbeddingCalculation.EmbeddingCalculation()
            embedding_calculation.calc_embed(train_set_indexed=train_set_indexed,
                                             val_set_indexed=val_set_indexed,
                                             batch_size=system_parameters['batch_size_embed'],
                                             vocab_chars=vocab_chars,
                                             vocab_lang=vocab_lang,
                                             max_context_window_size=system_parameters['max_context_window_size'],
                                             num_neg_samples=system_parameters['num_neg_samples'],
                                             max_eval_checks_not_improved=system_parameters['max_eval_checks_not_improved_embed'],
                                             max_num_epochs=system_parameters['max_num_epochs_embed'],
                                             eval_every_num_batches=system_parameters['eval_every_num_batches_embed'],
                                             lr_decay_every_num_batches=system_parameters['lr_decay_every_num_batches_embed'],
                                             lr_decay_factor=system_parameters['lr_decay_factor_embed'],
                                             initial_lr=system_parameters['initial_lr_embed'],
                                             embed_weights_rel_path=system_parameters['embed_weights_rel_path'],
                                             embed_model_checkpoint_rel_path=system_parameters['embed_model_checkpoint_rel_path'],
                                             system_param_dict=system_parameters,
                                             print_testing=system_parameters['print_embed_testing'],
                                             sampling_table_min_char_count=system_parameters['sampling_table_min_char_count'],
                                             sampling_table_specified_size_cap=system_parameters['sampling_table_specified_size_cap'])

    ################
    # RNN TRAINING #
    ################

        rnn_calculation = RNNCalculation.RNNCalculation(system_parameters)
        
        # train RNN model
        if (system_parameters['train_rnn']):
            rnn_calculation.train([train_set_indexed, val_set_indexed], vocab_lang, vocab_chars, system_parameters['rnn_model_checkpoint_rel_path'])

    ##############
    # EVALUATION #
    ##############

        # evaluate test set
        if (system_parameters['eval_test_set']):
            rnn_calculation.test([test_set_indexed], system_parameters['rnn_model_checkpoint_rel_path'])

        # print saved model checkpoint from file
        if (system_parameters['print_model_checkpoint'] != None and system_parameters['print_model_checkpoint_embed_weights'] != None):
            gru_model, _ = rnn_calculation.load_model_and_data([],system_parameters['print_model_checkpoint_embed_weights'])

            start_epoch, best_val_accuracy, test_accuracy, system_parameters, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(system_parameters['print_model_checkpoint'])

            to_print = [('Model:\n', gru_model),
                        ('Epochs trained: ', start_epoch),
                        ('Best validation set accuracy: ', best_val_accuracy),
                        ('Test set accuracy: ', test_accuracy),
                        ('Epochs trained: ', start_epoch),
                        ('System parameters used: ', system_parameters)]
            rnn_calculation.print_out(to_print)


if __name__ == '__main__':
    main()