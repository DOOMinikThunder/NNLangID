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
        	system_param_dict:
        """
        self.system_param_dict = system_param_dict

    """
    to run the terminal, only this function needs to be called
    """
    def run_terminal(self, can_use_live_tweets=False):
        """

        Args:
        	can_use_live_tweets:

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

        self.loop_input(gru_model=gru_model, input_data=input_data, can_use_live_tweets=can_use_live_tweets, embed=embed, vocab_lang=results_dict['vocab_lang'], vocab_chars=results_dict['vocab_chars'])


    def __loop_input(self, gru_model, input_data, can_use_live_tweets, embed, vocab_lang, vocab_chars):
        """

        Args:
        	gru_model:
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
            input_text, input_text_lang_tuple, is_live_tweets = self.retrieve_text(can_use_live_tweets, index2lang, tweet_retriever, vocab_lang)
            if input_text is None:
                continue
            input_text_embed_char_text_inp_tensors, _ = self.prepare_data(input_data=input_data,
                                                                          embed=embed,
                                                                          input_text_lang_tuple=input_text_lang_tuple,
                                                                          vocab_chars=vocab_chars,
                                                                          vocab_lang=vocab_lang)
            # transfer tensors to GPU if available
            if (self.system_param_dict['cuda_is_avail']):
                input_text_embed_char_text_inp_tensors = input_text_embed_char_text_inp_tensors.cuda()
            n_highest_probs = 5
            self.evaluate_and_print(gru_model=gru_model, input_text_embed_char_text_inp_tensors=input_text_embed_char_text_inp_tensors,
                                    n_highest_probs=n_highest_probs, input_text=input_text, index2lang=index2lang, is_live_tweets=is_live_tweets)

    def __str_to_int(self, string):
        """

        Args:
        	string:

        Returns:

        """
        try:
            number = int(string)
            return number
        except ValueError:
#            print("Not a number")
            return 0

    def sample_tweets(self, tweet_retriever, vocab_lang, amount):
        """

        Args:
        	tweet_retriever:
        	vocab_lang:
        	amount:

        Returns:

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

    def retrieve_text(self, can_use_live_tweets, index2lang, tweet_retriever, vocab_lang):
        """

        Args:
        	can_use_live_tweets:
        	index2lang:
        	tweet_retriever:
        	vocab_lang:

        Returns:

        """
        input_terminal = input('Enter text or number: ')
        amount_live_tweets = self.str_to_int(input_terminal)
        is_live_tweets = False
        if amount_live_tweets > 0:
            is_live_tweets = True
            sample_tweets = self.sample_tweets(tweet_retriever, vocab_lang, amount_live_tweets)
            print('sample_tweets',sample_tweets)
            if sample_tweets is None:
                return None, None
            input_text =  list(sample_tweets.values())
            input_text_lang_tuple = [(text, index2lang[0]) for text in input_text]
        else:
            input_text = [input_terminal]
            input_text_lang_tuple = [(input_text[0], index2lang[0])]  # language must be in vocab_lang
        return input_text, input_text_lang_tuple, is_live_tweets

    def prepare_data(self, input_data, embed, input_text_lang_tuple, vocab_chars, vocab_lang):
        """

        Args:
        	input_data:
        	embed:
        	input_text_lang_tuple:
        	vocab_chars:
        	vocab_lang:

        Returns:

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

    def evaluate_and_print(self, gru_model, input_text_embed_char_text_inp_tensors, n_highest_probs, input_text, index2lang, is_live_tweets):
        """

        Args:
        	gru_model:
        	input_text_embed_char_text_inp_tensors:
        	n_highest_probs:
        	input_text:
        	index2lang:
        	is_live_tweets:

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