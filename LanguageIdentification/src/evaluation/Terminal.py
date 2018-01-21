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


from input import DataSplit, InputData
from evaluation import RNNEvaluator
from net import GRUModel
try:
    from tweet_retriever import TweetRetriever
    can_use_tweets = True
except ImportError:
    can_use_tweets = False


class Terminal(object):
    
    def __init__(self, system_param_dict):
        """

        Args:
        	system_param_dict: contains hyperparameters
        """
        self.system_param_dict = system_param_dict

    """
    to run the terminal, only this function needs to be called
    """
    def run_terminal(self, can_use_live_tweets=False):
        """
        Enters an infinite loop to evaluate tweets entered manually or retrieved live from twitter
        Args:
        	can_use_live_tweets: true, if tweets can be retrieved from twitter

        Returns:

        """
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(self.system_param_dict['trained_embed_weights_rel_path'])
        gru_model = GRUModel.GRUModel(vocab_chars={},
                                      vocab_lang={},
                                      input_size=embed.weight.size()[1],    # equals embedding dimension
                                      num_classes=num_classes,
                                      system_param_dict=self.system_param_dict)
        state = gru_model.load_model_checkpoint_from_file(self.system_param_dict['trained_model_checkpoint_rel_path'])
        results_dict = state['results_dict']
        # run on GPU if available
        if (self.system_param_dict['cuda_is_avail']):
            gru_model.cuda()

        self.__loop_input(gru_model=gru_model, input_data=input_data, can_use_live_tweets=can_use_live_tweets, embed=embed, vocab_lang=results_dict['vocab_lang'], vocab_chars=results_dict['vocab_chars'])


    def __loop_input(self, gru_model, input_data, can_use_live_tweets, embed, vocab_lang, vocab_chars):
        """
        Takes user input and evaluates the resulting 'tweet'
        Args:
        	gru_model: model which will evaluate
        	input_data:
        	can_use_live_tweets:
        	embed:
        	vocab_lang:
        	vocab_chars:

        Returns:

        """
        tweet_retriever = TweetRetriever.TweetRetriever()
        lang2index, index2lang = input_data.get_string2index_and_index2string(vocab_lang)

        input_text = ''
        while input_text != 'exit':
            input_text, input_text_lang_tuple, is_live_tweets = self.__retrieve_text(can_use_live_tweets, index2lang, tweet_retriever, vocab_lang)
            if input_text is None:
                continue
            input_text_embed_char_text_inp_tensors, _ = self.__prepare_data(input_data=input_data,
                                                                          embed=embed,
                                                                          input_text_lang_tuple=input_text_lang_tuple,
                                                                          vocab_chars=vocab_chars,
                                                                          vocab_lang=vocab_lang)
            # transfer tensors to GPU if available
            if (self.system_param_dict['cuda_is_avail']):
                input_text_embed_char_text_inp_tensors = input_text_embed_char_text_inp_tensors.cuda()
            n_highest_probs = 5
            self.__evaluate_and_print(gru_model=gru_model, input_text_embed_char_text_inp_tensors=input_text_embed_char_text_inp_tensors,
                                    n_highest_probs=n_highest_probs, input_text=input_text, index2lang=index2lang, is_live_tweets=is_live_tweets)

    def __str_to_int(self, string):
        """
        Tries to cast a string to an integer
        Args:
        	string: will be cast to int

        Returns:
        	number: int(string), 0 otherwise

        """
        try:
            number = int(string)
            return number
        except ValueError:
#            print("Not a number")
            return 0

    def __sample_tweets(self, tweet_retriever, vocab_lang, amount):
        """
        calls tweet_retriever to get sample tweets from twitter
        Args:
        	tweet_retriever: instance of TweetRetriever that connects to twitter
        	vocab_lang: dict of languages
        	amount: amount of tweets to sample

        Returns:
        	sample tweets

        """
        track = input("(Optional) Specify a keyword to search in tweets: ")
        if track != "":
            language = input("(Optional) Specify a language identifier to search in tweets: ")
            if language in vocab_lang:
                return tweet_retriever.retrieve_specified_track_and_language(amount, track, language)
            else:
                print("ERROR: not a language identifier")
                return None
        else:
            return tweet_retriever.retrieve_sample_tweets(amount)

    def __retrieve_text(self, can_use_live_tweets, index2lang, tweet_retriever, vocab_lang):
        """
        Asks for user input and calls manual or twitter evaluation based on response
        Args:
        	can_use_live_tweets: true, if tweets can be retrieved from twitter
        	index2lang:
        	tweet_retriever: instance of TweetRetriever that connects to twitter
        	vocab_lang: dict of languages

        Returns:
        	input_text: manually entered phrase or tweets from twitter
        	input_text_lang_tuple: tuple of input_text and its language
        	is_live_tweets:

        """
        input_terminal = input('Enter text or number: ')
        amount_live_tweets = self.__str_to_int(input_terminal)
        is_live_tweets = False
        if amount_live_tweets > 0 and can_use_live_tweets:
            is_live_tweets = True
            sample_tweets = self.__sample_tweets(tweet_retriever, vocab_lang, amount_live_tweets)
            print('sample_tweets',sample_tweets)
            if sample_tweets is None:
                return None, None
            input_text =  list(sample_tweets.values())
            input_text_lang_tuple = [(text, index2lang[0]) for text in input_text]
        else:
            input_text = [input_terminal]
            input_text_lang_tuple = [(input_text[0], index2lang[0])]  # language must be in vocab_lang
        return input_text, input_text_lang_tuple, is_live_tweets

    def __prepare_data(self, input_data, embed, input_text_lang_tuple, vocab_chars, vocab_lang):
        """
        prepares input data to be fed in model
        Args:
        	input_data: instance of InputData class
        	embed: embedding needed to convert input data into char embedding
        	input_text_lang_tuple: actual input data
        	vocab_chars: dict of all characters learned
        	vocab_lang: dict of all languages

        Returns:
        	input_text_embed_char_text_inp_tensors:
        	input_text_target_tensors:

        """
        filtered_texts_and_lang = input_data.filter_out_irrelevant_tweet_parts(input_text_lang_tuple)
        # print('filtered_texts_and_lang',filtered_texts_and_lang)
        input_text_only_vocab_chars = input_data.get_texts_with_only_vocab_chars(filtered_texts_and_lang, vocab_chars)
        # print('input_text_only_vocab_chars', input_text_only_vocab_chars)
        input_text_indexed = input_data.get_indexed_texts_and_lang(input_text_only_vocab_chars, vocab_chars, vocab_lang)
        # print('input_text_indexed',input_text_indexed)
        input_text_embed_char_text_inp_tensors, input_text_target_tensors = input_data.create_embed_input_and_target_tensors \
            (indexed_texts_and_lang=input_text_indexed,
             embed_weights_rel_path=self.system_param_dict['trained_embed_weights_rel_path'],
             embed=embed)
        return input_text_embed_char_text_inp_tensors, input_text_target_tensors

    def __evaluate_and_print(self, gru_model, input_text_embed_char_text_inp_tensors, n_highest_probs, input_text, index2lang, is_live_tweets):
        """
        calls evaluator instance for prediction and prints languages with highest probabilites
        Args:
        	gru_model: model used for evaluation
        	input_text_embed_char_text_inp_tensors: input text as embedding for model
        	n_highest_probs: n highest probabilites of languages to return
        	input_text: actual input text
        	index2lang: lookup from unique index to language
        	is_live_tweets: true, if tweets from twitter are evaluated

        Returns:

        """
        rnn_evaluator = RNNEvaluator.RNNEvaluator(gru_model)
        for i, input_tensor in enumerate(input_text_embed_char_text_inp_tensors):
            lang_prediction, _ = rnn_evaluator.evaluate_single_date(input_tensor, n_highest_probs)
            if (is_live_tweets):
                print("====================\nTweet detected: \n\n%s\n" % input_text[i])

            # print n_highest_probs for input
            print('Language:')
            for i in range(len(lang_prediction)):
                if (i == 0):
                    print("{0:.2f}".format(lang_prediction[i][0] * 100) + "%: " + str(
                        index2lang[lang_prediction[i][1]]) + "\n")
                else:
                    print(
                        "{0:.2f}".format(lang_prediction[i][0] * 100) + "%: " + str(index2lang[lang_prediction[i][1]]))