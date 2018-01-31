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
    
    # USER PARAMETER
    # (only set this and the parameters in the corresponding YAML settings file;
    # True for SystemParametersCluster.yaml, False for SystemParameters.yaml)
    use_cluster_params = False
    
    
    ###########################################################################
    
    # automatically determined parameters
    if(use_cluster_params):
        yaml_file = 'SystemParametersCluster.yaml'
    else:
        yaml_file = 'SystemParameters.yaml'
    with open(yaml_file, 'r') as stream:
        system_param_dict = yaml.load(stream)
        
    system_param_dict['can_use_live_tweets'] = can_use_live_tweets
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
    
    # terminal for interactive testing on arbitrary input text or live tweets
    if (system_param_dict['run_terminal']):
        terminal = Terminal.Terminal(system_param_dict)
        terminal.run_terminal(system_param_dict['can_use_live_tweets'])
    else:

    ########################
    # DATA FILES SPLITTING #
    ########################

        # split into training, validation and test set files from an original file
        files_exist = Path(system_param_dict['out_tr_data_rel_path']).is_file() and Path(system_param_dict['out_va_data_rel_path']).is_file() and Path(system_param_dict['out_te_data_rel_path']).is_file()
        if (system_param_dict['create_splitted_data_files'] or not files_exist):
            out_filenames = [system_param_dict['out_tr_data_rel_path'], system_param_dict['out_va_data_rel_path'], system_param_dict['out_te_data_rel_path']] #same size as ratios
            data_splitter = DataSplit.DataSplit()
            splitted_data = data_splitter.split_percent_of_languages(input_file=system_param_dict['input_tr_va_te_data_rel_path'],
                                                                     ratio=system_param_dict['tr_va_te_split_ratios'],
                                                                     out_filenames=out_filenames,
                                                                     shuffle_seed=system_param_dict['split_shuffle_seed'])

    ##################################################
    # DATA RETRIEVAL, PREPROCESSING & TRANSFORMATION #
    ##################################################

        # retrieve, preprocess and transform data for readily use for embedding and RNN,
        # and get the vocabularies
        input_data = InputData.InputData()
        train_set_indexed, val_set_indexed, test_set_indexed, real_test_set_indexed, vocab_chars, vocab_lang = input_data.get_indexed_data(
            train_data_rel_path=system_param_dict['out_tr_data_rel_path'],
            validation_data_rel_path=system_param_dict['out_va_data_rel_path'],
            test_data_rel_path=system_param_dict['out_te_data_rel_path'],
            real_test_data_rel_path=system_param_dict['input_rt_data_rel_path'],
            min_char_frequency=system_param_dict['min_char_frequency'],
            fetch_only_langs=system_param_dict['fetch_only_langs'],
            fetch_only_first_x_tweets=system_param_dict['fetch_only_first_x_tweets'])

    ######################
    # EMBEDDING TRAINING #
    ######################

        # train character embedding (Skip-Gram with Negative Sampling) to get embedding weights;
        # the loss-based best embedding model checkpoint and its weights are saved to file
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
        
        # train RNN model (uni- or bidirectional GRU) with character embeddings;
        # the loss-based best model checkpoint is saved to file
        if (system_param_dict['train_rnn']):
            rnn_calculation.train_rnn(data_sets=[train_set_indexed, val_set_indexed],
                                      vocab_chars=vocab_chars,
                                      vocab_lang=vocab_lang)

    ##############
    # EVALUATION #
    ##############

        # evaluate RNN model checkpoint on test set
        if (system_param_dict['eval_test_set']):
            rnn_calculation.test_rnn(data_sets=[test_set_indexed],
                                     vocab_chars=vocab_chars,
                                     vocab_lang=vocab_lang)

        # print saved model checkpoint from file (RNN or embedding)
        # note: some parameters in the YAML settings file have to be the same as in the checkpoint
        # (e.g. input_tr_va_te_data_rel_path and hidden_size_rnn)
        if (system_param_dict['print_model_checkpoint_embed_weights'] != None and system_param_dict['print_rnn_model_checkpoint'] != None):
            rnn_calculation.print_model_checkpoint(vocab_chars=vocab_chars,
                                                   vocab_lang=vocab_lang,
                                                   is_rnn_model=True)
        elif (system_param_dict['print_model_checkpoint_embed_weights'] != None and system_param_dict['print_embed_model_checkpoint'] != None):
            rnn_calculation.print_model_checkpoint(vocab_chars=vocab_chars,
                                                   vocab_lang=vocab_lang,
                                                   is_rnn_model=False)


if __name__ == '__main__':
    main()