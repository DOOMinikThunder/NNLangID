# -*- coding: utf-8 -*-

import math
from pathlib import Path
import torch
import Evaluator
from input import DataSplit, InputData
from embedding import EmbeddingCalculation
from net import GRUModel



def main():

    
    ##############
    # PARAMETERS #
    ##############


    # SYSTEM
    create_splitted_data_files = True                     # split into training, validation and test set from an original file
    calc_embed = True
    train_rnn = True
    eval_test_set = True
    
    print_embed_testing = False
    print_model_checkpoint_embed_weights = None#"../data/embed_weights/trained/embed_weights_de_en_es_fr_it_und.txt"#None
    print_model_checkpoint = None#"../data/model_checkpoints/trained/model_checkpoint_de_en_es_fr_it_und.pth"#None
    
    terminal = True                                      # if True: disables all other calculations
    
    
    # DATA
    input_tr_va_te_data_rel_path = "../data/input_data/original/recall_oriented_dl.csv" #training, validation and test will be generated from this file
#    input_tr_va_te_data_rel_path = "../data/input_data/original/uniformly_sampled_dl.csv" #training, validation and test will be generated from this file
#    input_tr_va_te_data_rel_path = "../data/input_data/testing/test_embed.csv" #training, validation and test will be generated from this file
#    input_tr_va_te_data_rel_path = "../data/input_data/testing/test_recall_de_en_es.csv" #training, validation and test will be generated from this file
    input_rt_data_rel_path = "../data/input_data/original/uniformly_sampled_dl.csv" #to change later, rt = real test
    
    embed_weights_rel_path = "../data/save/embed_weights.txt"
#    trained_embed_weights_rel_path = "../data/save/embed_weights.txt"
    trained_embed_weights_rel_path = "../data/save/trained/embed_weights_de_en_es_fr_it.txt"
    model_checkpoint_rel_path = "../data/save/model_checkpoint.pth"
#    trained_model_checkpoint_rel_path = "../data/save/model_checkpoint.pth"
    trained_model_checkpoint_rel_path = "../data/save/trained/model_checkpoint_de_en_es_fr_it.pth"
    
    tr_va_te_split_ratios = [0.8, 0.1, 0.1]                  # [train_ratio, val_ratio, test_ratio]
    split_shuffle_seed = 42                                  # ensures that splitted sets (training, validation, test) are always created identically (given a specified ratio)
    fetch_only_langs = ['de', 'en', 'es', 'fr', 'it']#['de', 'en', 'es']#['el', 'fa', 'hi', 'ca']#None
    fetch_only_first_x_tweets = math.inf#5
    min_char_frequency = 2                                   # chars appearing less than min_char_frequency in the training set will not be used to create the vocabulary vocab_chars
    
    
    # HYPERPARAMETERS EMBEDDING
    sampling_table_min_char_count = 10                       # determines the precision of the sampling (should be 10 or higher)
    sampling_table_specified_size_cap = 10000#math.inf        # caps specified sampling table size to this value (no matter how big it would be according to sampling_table_min_char_count)
                                                             # note: this is only the specified size, the actual table size may slightly deviate due to roundings in the calculation
#    embed_dim = 2                                           # will be set automatically later to: roundup(log2(vocabulary-size))
    max_context_window_size = 2
    num_neg_samples = 5
    batch_size_embed = 10
    initial_lr_embed = 0.025
    lr_decay_num_batches_embed = 100
    num_epochs_embed = 1
   
    
    # HYPERPARAMETERS RNN
#    input_size = list(train_embed_char_text_inp_tensors[0].size())[2]
#    num_classes = len(vocab_lang)
    hidden_size_rnn = 100
    num_layers_rnn = 1
    is_bidirectional = True
    batch_size_rnn = 10
    initial_lr_rnn = 0.01
    scheduler_step_size_rnn = 1                              # currently not functioning
    scheduler_gamma_rnn = 0.1                                # currently not functioning
    weight_decay_rnn = 0.00001
    num_epochs_rnn = 10#math.inf#2

    
    # set dict to later store parameters to file
    system_param_dict = {
        # SYSTEM
        'create_splitted_data_files' : create_splitted_data_files,
        'calc_embed' : calc_embed,
        'train_rnn' : train_rnn,
        'eval_test_set' : eval_test_set,
        'print_embed_testing': print_embed_testing,
        'print_model_checkpoint_embed_weights' : print_model_checkpoint_embed_weights,
        'print_model_checkpoint' : print_model_checkpoint,
        'terminal' : terminal,
        # DATA
        'input_tr_va_te_data_rel_path' : input_tr_va_te_data_rel_path,
        'input_rt_data_rel_path' : input_rt_data_rel_path,
        'embed_weights_rel_path' : embed_weights_rel_path,
        'trained_embed_weights_rel_path' : trained_embed_weights_rel_path,
        'model_checkpoint_rel_path' : model_checkpoint_rel_path,
        'trained_model_checkpoint_rel_path' : trained_model_checkpoint_rel_path,
        'tr_va_te_split_ratios' : tr_va_te_split_ratios,
        'split_shuffle_seed' : split_shuffle_seed,
        'fetch_only_langs' : fetch_only_langs,
        'fetch_only_first_x_tweets' : fetch_only_first_x_tweets,
        'min_char_frequency' : min_char_frequency,
        # HYPERPARAMETERS EMBEDDING
        'sampling_table_min_char_count' : sampling_table_min_char_count,
        'sampling_table_specified_size_cap' : sampling_table_specified_size_cap,
        'max_context_window_size' : max_context_window_size,
        'num_neg_samples' : num_neg_samples,
        'batch_size_embed' : batch_size_embed,
        'initial_lr_embed' : initial_lr_embed,
        'lr_decay_num_batches_embed' : lr_decay_num_batches_embed,
        'num_epochs_embed' : num_epochs_embed,
        # HYPERPARAMETERS RNN
        'hidden_size_rnn' : hidden_size_rnn,
        'num_layers_rnn' : num_layers_rnn,
        'is_bidirectional' : is_bidirectional,
        'batch_size_rnn' : batch_size_rnn,
        'initial_lr_rnn' : initial_lr_rnn,
        'scheduler_step_size_rnn' : scheduler_step_size_rnn,
        'scheduler_gamma_rnn' : scheduler_gamma_rnn,
        'weight_decay_rnn' : weight_decay_rnn,
        'num_epochs_rnn' : num_epochs_rnn,
        }
    
    
    cuda_is_avail = torch.cuda.is_available()
    cuda_is_avail = False #cuda not working on some builds
    print('cuda on') if cuda_is_avail else print('cuda off')
    
 
    ############
    # TERMINAL #
    ############    
    
    # simple terminal for testing
    if (terminal):
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(trained_embed_weights_rel_path)
#        print(embed.weight.size()[1])
        gru_model = GRUModel.GRUModel(input_size=embed.weight.size()[1],    # equals embedding dimension
                                      hidden_size=hidden_size_rnn,
                                      num_layers=num_layers_rnn,
                                      num_classes=num_classes,
                                      is_bidirectional=is_bidirectional,
                                      initial_lr=initial_lr_rnn,
                                      weight_decay=weight_decay_rnn)
        start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(trained_model_checkpoint_rel_path)
        # run on GPU if available
        if (cuda_is_avail):
            gru_model.cuda()
#        print('Model:\n', gru_model)
        evaluator = Evaluator.Evaluator(gru_model)
        lang2index, index2lang = input_data.get_string2index_and_index2string(vocab_lang)
        
        input_text = ''
        while input_text != 'exit':
            input_text = input('Enter text: ')
#            print(input_text)
            input_text_lang_tuple = [(input_text, index2lang[0])]    # language must be in vocab_lang
            
            filtered_texts_and_lang = input_data.filter_out_irrelevant_tweet_parts(input_text_lang_tuple)
#            print(filtered_texts_and_lang)
            input_text_only_vocab_chars = input_data.get_texts_with_only_vocab_chars(filtered_texts_and_lang, vocab_chars)
#            print(input_text_only_vocab_chars)
            input_text_indexed = input_data.get_indexed_texts_and_lang(input_text_only_vocab_chars, vocab_chars, vocab_lang)
#            print(input_text_indexed)
            input_text_embed_char_text_inp_tensors, input_text_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=input_text_indexed,
                                                                                                                                 embed_weights_rel_path=trained_embed_weights_rel_path)
            # transfer tensors to GPU if available
            if (cuda_is_avail):
                input_text_embed_char_text_inp_tensors = input_text_embed_char_text_inp_tensors.cuda()
                input_text_target_tensors = input_text_target_tensors.cuda()
            
            n_highest_probs = 5
            lang_prediction = evaluator.evaluate_single_date(input_text_embed_char_text_inp_tensors[0],
                                                             n_highest_probs)
            # print n_highest_probs for input
            print('Language:')
            for i in range(len(lang_prediction)):
                if (i == 0):
                    print(lang_prediction[i][0], ':', index2lang[lang_prediction[i][1]])
                    print('')
                else:
                    print(lang_prediction[i][0], ':', index2lang[lang_prediction[i][1]])
    
        
    ########################
    # DATA FILES SPLITTING #
    ########################
    
    if (not terminal):
        # split into training, validation and test set from an original file
        out_tr_data_rel_path = "../data/input_data/original_splitted/training.csv"
        out_va_data_rel_path = "../data/input_data/original_splitted/validation.csv"
        out_te_data_rel_path = "../data/input_data/original_splitted/test.csv"
        files_exist = Path(out_tr_data_rel_path).is_file() and Path(out_va_data_rel_path).is_file() and Path(out_te_data_rel_path).is_file()
        if (create_splitted_data_files or not files_exist):
            out_filenames = [out_tr_data_rel_path, out_va_data_rel_path, out_te_data_rel_path] #same size as ratios
            data_splitter = DataSplit.DataSplit()
            splitted_data = data_splitter.split_percent_of_languages(input_tr_va_te_data_rel_path, tr_va_te_split_ratios, out_filenames, split_shuffle_seed)
    

    ###################################
    # DATA RETRIEVAL & TRANSFORMATION #
    ###################################

    if (not terminal):
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
            train_data_rel_path=out_tr_data_rel_path,
            validation_data_rel_path=out_va_data_rel_path,
            test_data_rel_path=out_te_data_rel_path,
            real_test_data_rel_path=input_rt_data_rel_path,
            min_char_frequency=min_char_frequency,
            fetch_only_langs=fetch_only_langs,
            fetch_only_first_x_tweets=fetch_only_first_x_tweets)
#        print(train_set_indexed, val_set_indexed, test_set_indexed)
#        print(vocab_chars)
#        print(len(vocab_chars))
#        print(vocab_lang)
#        print(len(vocab_lang))


    #########################
    # EMBEDDING CALCULATION #
    #########################
    
    if (not terminal and calc_embed):
        indexed_texts = input_data.get_only_indexed_texts(train_set_indexed)
#        print(indexed_texts)
        embedding_calculation = EmbeddingCalculation.EmbeddingCalculation()
        embedding_calculation.calc_embed(indexed_tweet_texts=indexed_texts,
                                         batch_size=batch_size_embed,
                                         vocab_chars=vocab_chars,
                                         vocab_lang=vocab_lang,
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

    # train RNN model
    if (not terminal and train_rnn):
        train_embed_char_text_inp_tensors, train_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=train_set_indexed,
                                                                                                                   embed_weights_rel_path=embed_weights_rel_path)
        val_embed_char_text_inp_tensors, val_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=val_set_indexed,
                                                                                                               embed_weights_rel_path=embed_weights_rel_path)
        # transfer tensors to GPU if available
        if (cuda_is_avail):
            train_embed_char_text_inp_tensors = [tensor.cuda() for tensor in train_embed_char_text_inp_tensors]
            train_target_tensors = [tensor.cuda() for tensor in train_target_tensors]
            val_embed_char_text_inp_tensors = [tensor.cuda() for tensor in val_embed_char_text_inp_tensors]
            val_target_tensors = [tensor.cuda() for tensor in val_target_tensors]

        gru_model = GRUModel.GRUModel(input_size=list(train_embed_char_text_inp_tensors[0].size())[2],  # equals embedding dimension
                                      hidden_size=hidden_size_rnn,
                                      num_layers=num_layers_rnn,
                                      num_classes=len(vocab_lang),
                                      is_bidirectional=is_bidirectional,
                                      initial_lr=initial_lr_rnn,
                                      weight_decay=weight_decay_rnn)
        # run on GPU if available
        if (cuda_is_avail):
            gru_model.cuda()
        print('Model:\n', gru_model)
        evaluator = Evaluator.Evaluator(gru_model)
        
        cur_val_accuracy = 0
        best_val_accuracy = -1.0

        epoch = 0
        is_improving = True
        # stop training when validation set error stops getting smaller ==> stop when overfitting occurs
        # or when maximum number of epochs reached
        while is_improving and not epoch == num_epochs_rnn:
            print('RNN epoch:', epoch)
            # inputs: whole data set, every date contains the embedding of one char in one dimension
            # targets: whole target set, target is set for each character embedding
            gru_model.train(inputs=train_embed_char_text_inp_tensors,
                            targets=train_target_tensors,
                            batch_size=batch_size_rnn)
            
            # evaluate validation set
            cur_val_accuracy, val_conf_matrix = evaluator.evaluate_data_set(val_embed_char_text_inp_tensors,
                                                           val_target_tensors,
                                                           vocab_lang)
            print('========================================')
            print('confusion matrix:\n')
            print(evaluator.to_string_confusion_matrix(confusion_matrix=val_conf_matrix, vocab_lang=vocab_lang, pad=5))
            print('========================================')
            print('Epoch', epoch, 'validation set accuracy:', cur_val_accuracy)
            print('========================================')
            
            # check if accuracy improved and if so, save model checkpoint to file
            if (best_val_accuracy < cur_val_accuracy):
                best_val_accuracy = cur_val_accuracy
                gru_model.save_model_checkpoint_to_file({
                                            'start_epoch': epoch + 1,
                                            'best_val_accuracy': best_val_accuracy,
                                            'test_accuracy' : -1.0,
                                            'state_dict': gru_model.state_dict(),
                                            'optimizer': gru_model.optimizer.state_dict(),
                                            'system_param_dict' : system_param_dict,
                                            'vocab_chars' : vocab_chars,
                                            'vocab_lang' : vocab_lang,
                                            },
                                            model_checkpoint_rel_path)
            else:
                is_improving = False
            epoch += 1
           
            
    ##############
    # EVALUATION #
    ##############
    
    # evaluate test set
    if (not terminal and eval_test_set):
        test_embed_char_text_inp_tensors, test_target_tensors = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=test_set_indexed,
                                                                                                                 embed_weights_rel_path=embed_weights_rel_path)
        # transfer tensors to GPU if available
        if (cuda_is_avail):
            test_embed_char_text_inp_tensors = test_embed_char_text_inp_tensors.cuda()
            test_target_tensors = test_target_tensors.cuda()
            
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(embed_weights_rel_path)
        gru_model = GRUModel.GRUModel(input_size=embed.weight.size()[1],    # equals embedding dimension
                                      hidden_size=hidden_size_rnn,
                                      num_layers=num_layers_rnn,
                                      num_classes=num_classes,
                                      is_bidirectional=is_bidirectional,
                                      initial_lr=initial_lr_rnn,
                                      weight_decay=weight_decay_rnn)
        start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(model_checkpoint_rel_path)
        # run on GPU if available
        if (cuda_is_avail):
            gru_model.cuda()
#        print('Model:\n', gru_model)
        evaluator = Evaluator.Evaluator(gru_model)
        
        test_accuracy, test_conf_matrix = evaluator.evaluate_data_set(test_embed_char_text_inp_tensors,
                                                                 test_target_tensors,
                                                                 vocab_lang)
        print('========================================')
        print('Epochs trained:', start_epoch)
        print('========================================')
        print('Best validation set accuracy:', best_val_accuracy)
        print('========================================')
        print('Test set accuracy:', test_accuracy)
        print('========================================')
        print('System parameters used:')
        for param in system_param_dict:
            print(repr(param), ':', system_param_dict[param])
        print('========================================')

        # save test_accuracy to file
        gru_model.save_model_checkpoint_to_file({
                                            'start_epoch': start_epoch,
                                            'best_val_accuracy': best_val_accuracy,
                                            'test_accuracy' : test_accuracy,
                                            'state_dict': gru_model.state_dict(),
                                            'optimizer': gru_model.optimizer.state_dict(),
                                            'system_param_dict' : system_param_dict,
                                            'vocab_chars' : vocab_chars,
                                            'vocab_lang' : vocab_lang,
                                            },
                                            model_checkpoint_rel_path)


    # print saved model checkpoint from file
    if (not terminal and print_model_checkpoint != None and print_model_checkpoint_embed_weights != None):
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(print_model_checkpoint_embed_weights)
        gru_model = GRUModel.GRUModel(input_size=embed.weight.size()[1],    # equals embedding dimension
                                      hidden_size=hidden_size_rnn,
                                      num_layers=num_layers_rnn,
                                      num_classes=num_classes,
                                      is_bidirectional=is_bidirectional,
                                      initial_lr=initial_lr_rnn,
                                      weight_decay=weight_decay_rnn)
#        print('Model:\n', gru_model)
        start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(print_model_checkpoint)
        print('========================================')
        print('Epochs trained:', start_epoch)
        print('========================================')
        print('Best validation set accuracy:', best_val_accuracy)
        print('========================================')
        print('Test set accuracy:', test_accuracy)
        print('========================================')
        print('System parameters used:')
        for param in system_param_dict:
            print(repr(param), ':', system_param_dict[param])
        print('========================================')
    
    


if __name__ == '__main__':
    main()