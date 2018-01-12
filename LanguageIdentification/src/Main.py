# -*- coding: utf-8 -*-

import math
import InputData
import Evaluator
from embedding import EmbeddingCalculation
from net import GRUModel



def main():

    ##############
    # PARAMETERS #
    ##############
    
#    input_data_rel_path = "../data/input_data/recall_oriented_dl.csv"
    input_data_rel_path = "../data/input_data/uniformly_sampled_dl.csv"
#    input_data_rel_path = "../data/input_data/test_embed.csv"
    test_data_rel_path = "../data/input_data/uniformly_sampled_dl.csv"
#    test_data_rel_path = "../data/input_data/test_embed.csv"

    embed_weights_rel_path = "../data/embed_weights/embed_weights.txt"
    val_model_checkpoint_rel_path = "../data/model_checkpoints/val_model_checkpoint.pth"
    test_model_checkpoint_rel_path = "../data/model_checkpoints/test_model_checkpoint.pth"
    fetch_only_langs = ['pl', 'sv']#['el', 'fa', 'hi', 'ca']#None
    fetch_only_first_x_tweets = math.inf#5
    calc_embed = True
    train_rnn = True
    eval_test_set = True
    print_embed_testing = False
    print_model_checkpoints = False
    
    # HYPERPARAMETERS EMBEDDING
    set_ratios = [0.8, 0.2]                                  # [train_ratio, val_ratio]
                                                             # warning: changes may require new embedding calculation due to differently shuffled train_set
    min_char_frequency = 2
    sampling_table_min_char_count = 10                       # determines the precision of the sampling (should be 10 or higher)
    sampling_table_specified_size_cap = 1000#math.inf        # caps specified sampling table size to this value (no matter how big it would be according to sampling_table_min_char_count)
                                                             # note: this is only the specified size, the actual table size may slightly deviate due to roundings in the calculation
    batch_size_embed = 2
    max_context_window_size = 2
    num_neg_samples = 5
#    embed_dim = 2                                           # will be set automatically later to: roundup(log2(vocabulary-size))
    initial_lr_embed = 0.025
    lr_decay_num_batches_embed = 100
    num_epochs_embed = 1

    # HYPERPARAMETERS RNN
#    input_size = list(train_embed_char_text_inp_tensors[0].size())[2]
#    num_classes = len(vocab_lang)
    hidden_size = 100
    num_layers = 1
    is_bidirectional = True
    initial_lr_rnn = 0.001
    scheduler_step_size_rnn = 1                              # currently not functioning
    scheduler_gamma_rnn = 0.1                                # currently not functioning
    weight_decay_rnn = 0.00001
    num_epochs_rnn = 10#math.inf#2
    batch_size_rnn = 5
    
    
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
                                                                                                                test_data_rel_path=test_data_rel_path,
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
                                         num_epochs=num_epochs_embed,
                                         initial_lr=initial_lr_embed,
                                         lr_decay_num_batches=lr_decay_num_batches_embed,
                                         embed_weights_rel_path=embed_weights_rel_path,
                                         print_testing=print_embed_testing,
                                         sampling_table_min_char_count=sampling_table_min_char_count,
                                         sampling_table_specified_size_cap=sampling_table_specified_size_cap)
    
    
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
                                  is_bidirectional=is_bidirectional,
                                  initial_lr=initial_lr_rnn,
                                  weight_decay=weight_decay_rnn,
                                  batch_size=batch_size_rnn)
    print('Model:\n', gru_model)
    evaluator = Evaluator.Evaluator(gru_model)


    # TRAINING
    if (train_rnn):
        cur_accuracy = 0
        best_accuracy = 0
        epoch = 0
        is_improving = True
        # stop training when validation set error stops getting smaller ==> stop when overfitting occurs
        # or when maximum number of epochs reached
        while is_improving and not epoch == num_epochs_rnn:
            print('RNN epoch:', epoch)
            # inputs: whole data set, every date contains the embedding of one char in one dimension
            # targets: whole target set, target is set for each character embedding
            gru_model.train(inputs=train_embed_char_text_inp_tensors,
                            targets=train_target_tensors)
            
            # evaluate validation set
            cur_accuracy = evaluator.evalute_data_set(val_embed_char_text_inp_tensors,
                                                      val_target_tensors,
                                                      vocab_lang)
            print('========================================')
            print('Epoch', epoch, 'validation set accuracy:', cur_accuracy)
            print('========================================')
            
            # check if accuracy improved and if so, save model checkpoint to file
            if (best_accuracy < cur_accuracy):
                best_accuracy = cur_accuracy
                gru_model.save_model_checkpoint_to_file({
                                            'start_epoch': epoch + 1,
                                            'state_dict': gru_model.state_dict(),
                                            'best_accuracy': best_accuracy,
                                            'optimizer': gru_model.optimizer.state_dict()
                                            },
                                            val_model_checkpoint_rel_path)
            else:
                is_improving = False
            epoch += 1
            
        
    # EVALUATION
    if (eval_test_set):
        # evaluate test set
        start_epoch, val_accuracy = gru_model.load_model_checkpoint_from_file(val_model_checkpoint_rel_path)
#        print(start_epoch)
#        print(best_accuracy)
        test_accuracy = evaluator.evalute_data_set(test_embed_char_text_inp_tensors,
                                                   test_target_tensors,
                                                   vocab_lang)
        print('========================================')
        print('Epochs trained:', start_epoch)
        print('========================================')
        print('Best validation set accuracy:', val_accuracy)
        print('========================================')
        print('Test set accuracy:', test_accuracy)
        print('========================================')

        # save test_accuracy to file
        gru_model.save_model_checkpoint_to_file({
                                            'start_epoch': start_epoch,
                                            'state_dict': gru_model.state_dict(),
                                            'best_accuracy': test_accuracy,
                                            'optimizer': gru_model.optimizer.state_dict()
                                            },
                                            test_model_checkpoint_rel_path)

    # print saved model checkpoints from file
    if (print_model_checkpoints):
        start_epoch, val_accuracy = gru_model.load_model_checkpoint_from_file(val_model_checkpoint_rel_path)
        start_epoch, test_accuracy = gru_model.load_model_checkpoint_from_file(test_model_checkpoint_rel_path)
        print('========================================')
        print('Epochs trained:', start_epoch)
        print('========================================')
        print('Best validation set accuracy:', val_accuracy)
        print('========================================')
        print('Test set accuracy:', test_accuracy)
        print('========================================')
        
    
    


if __name__ == '__main__':
    main()