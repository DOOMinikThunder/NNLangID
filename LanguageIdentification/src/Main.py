# -*- coding: utf-8 -*-

import math
import InputData
from embedding import EmbeddingCalculation
from net import GRUModel



def main():
    
    ##############
    # PARAMETERS #
    ##############
    
    input_data_rel_path = "../data/input_data/uniformly_sampled_dl.csv"
#    input_data_rel_path = "../data/input_data/test.csv"
    embed_weights_rel_path = "../data/embed_weights/embed_weights.txt"
    model_checkpoint_rel_path = "../data/model_checkpoint/model_checkpoint.pth"
    fetch_only_langs = ['pl', 'sv']#['el', 'fa', 'hi', 'ca']#None
    fetch_only_first_x_tweets = math.inf#5
    calc_embed = False
    
    # HYPERPARAMETERS EMBEDDING
    set_ratios = [1.0, 0.0, 0.0]    # [train_ratio, val_ratio, test_ratio]
                                    # warning: changes may require new embedding calculation due to differently shuffled train_set
    min_char_frequency = 2
    sampling_table_size = 1000
    batch_size_embed = 2
    max_context_window_size = 2
    num_neg_samples = 5
#    embed_dim = 2   # will be set automatically later
    initial_lr_embed = 0.025
    num_epochs_embed = 2
    
    # HYPERPARAMETERS RNN
#    input_size = list(train_embed_char_text_inp_tensors[0].size())[2]
#    num_classes = len(vocab_lang)
    hidden_size = 100
    num_layers = 1
    is_bidirectional = True
    
    
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
    train_set_indexed, val_set_indexed, test_set_indexed, vocab_chars, vocab_lang = input_data.get_indexed_data(input_data_rel_path=input_data_rel_path,
                                                                                                                min_char_frequency=min_char_frequency,
                                                                                                                set_ratios=set_ratios,
                                                                                                                fetch_only_langs=fetch_only_langs,
                                                                                                                fetch_only_first_x_tweets=fetch_only_first_x_tweets)
#    print(train_set_indexed, val_set_indexed, test_set_indexed)
#    print(vocab_chars)
#    print(vocab_lang)
    

    #########################
    # EMBEDDING CALCULATION #
    #########################
    
    if (calc_embed):
        indexed_texts = input_data.get_only_indexed_texts(train_set_indexed)
#        print(indexed_texts)
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
 
    # INITIALIZATION
    train_embed_char_text_inp_tensors, train_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=train_set_indexed,
                                                                                                               embed_weights_rel_path=embed_weights_rel_path)
    val_embed_char_text_inp_tensors, val_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=val_set_indexed,
                                                                                                           embed_weights_rel_path=embed_weights_rel_path)
    test_embed_char_text_inp_tensors, test_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=test_set_indexed,
                                                                                                             embed_weights_rel_path=embed_weights_rel_path)
    gru_model = GRUModel.GRUModel(input_size=list(train_embed_char_text_inp_tensors[0].size())[2],
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  num_classes=len(vocab_lang),
                                  is_bidirectional=is_bidirectional)
    print('Model:\n', gru_model)


    # TRAINING
    cur_accuracy = 0
    best_accuracy = 0
    epoch = 0
    is_improving = True
    # stop training when validation set error stops getting smaller ==> stop when overfitting occurs
    while is_improving:
        print('RNN epoch:', epoch)
        # inputs: whole data set, every date contains the embedding of one char in one dimension
        # targets: whole target set, target is set for each character embedding
        gru_model.train(inputs=train_embed_char_text_inp_tensors,
                        targets=train_target_tensors)
        
#        # evaluate validation set
#        cur_accuracy = gru_model.train(inputs=val_embed_char_text_inp_tensors,
#                                       targets=val_target_tensors,
#                                       eval=True)
        
        # evaluate train set
        cur_accuracy = gru_model.train(inputs=train_embed_char_text_inp_tensors,
                                       targets=train_target_tensors,
                                       eval=True)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        print('EVAL ACCURACY:', cur_accuracy)
        
        # check if accuracy improved and if so, save model checkpoint to file
        if (best_accuracy < cur_accuracy):
            best_accuracy = cur_accuracy
            gru_model.save_model_checkpoint_to_file({
                                        'start_epoch': epoch + 1,
                                        'state_dict': gru_model.state_dict(),
                                        'best_accuracy': best_accuracy,
                                        'optimizer': gru_model.optimizer.state_dict()
                                        },
                                        model_checkpoint_rel_path)
        else:
            is_improving = False
        epoch += 1
        
        
    # EVALUATION
#    # evaluate test set
#    start_epoch, best_accuracy = gru_model.load_model_checkpoint_from_file(model_checkpoint_rel_path)
#    accuracy = gru_model.train(inputs=test_embed_char_text_inp_tensors,
#                               targets=test_target_tensors,
#                               eval=True)
#    print('TEST ACCURACY:\n', accuracy)

    # evaluate train set
    start_epoch, best_accuracy = gru_model.load_model_checkpoint_from_file(model_checkpoint_rel_path)
#    print(start_epoch)
#    print(best_accuracy)
    cur_accuracy = gru_model.train(inputs=train_embed_char_text_inp_tensors,
                                   targets=train_target_tensors,
                                   eval=True)
    print('FINAL EVAL ACCURACY:', cur_accuracy)

    
    

if __name__ == '__main__':
    main()