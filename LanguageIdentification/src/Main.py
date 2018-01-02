# -*- coding: utf-8 -*-



def main():
    
    ##############
    # PARAMETERS #
    ##############
    
#    input_data_rel_path = "../data/input_data/uniformly_sampled_dl.csv"
    input_data_rel_path = "../data/input_data/test.csv"
    embed_weights_rel_path = "../data/embed_weights/embed_weights.txt"
    fetch_only_lang_x = None#'de'
    fetch_only_first_x_tweets = math.inf#5
    
    # Hyperparameters
    min_char_frequency = 2
    sampling_table_size = 1000
    batch_size = 2
    max_context_window_size = 2
    num_neg_samples = 5
#    embed_dim = 2   # will be set automatically later
    initial_lr = 0.025
    num_epochs = 1
    
    
    ###################################
    # DATA RETRIEVAL & TRANSFORMATION #
    ###################################
    
    input_data = InputData()
    indexed_texts_and_lang, vocab_chars, vocab_lang = input_data.get_indexed_data(input_data_rel_path,
                                                                                  min_char_frequency,
                                                                                  fetch_only_lang_x,
                                                                                  fetch_only_first_x_tweets)
#    print(indexed_texts_and_lang)
#    print(vocab_chars)
#    print(vocab_lang)
    
    # get only tweet texts for embedding
    indexed_tweet_texts = []
    for i in range(len(indexed_texts_and_lang)):
        indexed_tweet_texts.append(indexed_texts_and_lang[i][0])
#    print(indexed_tweet_texts)
    
    
    #########################
    # EMBEDDING CALCULATION #
    #########################
    
    embedding_calculation = EmbeddingCalculation()
    embedding_calculation.calc_embed(indexed_tweet_texts,
                                     batch_size,
                                     vocab_chars,
                                     max_context_window_size,
                                     num_neg_samples,
                                     sampling_table_size,
                                     num_epochs,
                                     initial_lr,
                                     embed_weights_rel_path)
    
    
    
    
    
    
    
    
    
    
    
    
        
#    context_target_onehot = embedding_calculation.create_context_target_onehot_vectors(2, tweet_texts_only_embed_chars, chars_for_embed)
#    print(len(context_target_onehot[0][0]))
#    print(len(context_target_onehot))
#    print(context_target_onehot)
    
# int to onehot conversion
#    b = np.zeros((a.size, a.max()+1))
#    b[np.arange(a.size),a] = 1
    


if __name__ == '__main__':
    main()