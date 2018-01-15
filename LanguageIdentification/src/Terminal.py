from input import DataSplit, InputData
from evaluation import Evaluator
from net import GRUModel
try:
	from tweet_retriever import TweetRetriever
	can_use_tweets = True
except ImportError:
	can_use_tweets = False

class Terminal(object):
	def __init__(self, system_parameters):
		self.system_parameters = system_parameters

	def use_terminal(self, terminal_live_tweets=False):
		input_data = InputData.InputData()
		embed, num_classes = input_data.create_embed_from_weights_file(self.system_parameters['trained_embed_weights_rel_path'])
		gru_model = GRUModel.GRUModel(input_size=embed.weight.size()[1],    # equals embedding dimension
									  hidden_size=self.system_parameters['hidden_size_rnn'],
									  num_layers=self.system_parameters['num_layers_rnn'],
									  num_classes=num_classes,
									  is_bidirectional=self.system_parameters['is_bidirectional'],
									  initial_lr=self.system_parameters['initial_lr_rnn'],
									  weight_decay=self.system_parameters['weight_decay_rnn'])
		start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file \
			(self.system_parameters['trained_model_checkpoint_rel_path'])
		# run on GPU if available
		if (self.system_parameters['cuda_is_avail']):
			gru_model.cuda()

		self.loop_input(gru_model=gru_model, input_data=input_data, terminal_live_tweets=terminal_live_tweets, embed=embed, vocab_lang=vocab_lang, vocab_chars=vocab_chars)


	def loop_input(self, gru_model, input_data, terminal_live_tweets, embed, vocab_lang, vocab_chars):
		tweet_retriever = TweetRetriever.TweetRetriever()
		lang2index, index2lang = input_data.get_string2index_and_index2string(vocab_lang)

		input_text = ''
		while input_text != 'exit':
			input_text, input_text_lang_tuple = self.retrieve_text(terminal_live_tweets, index2lang, tweet_retriever, vocab_lang)
			if input_text is None:
				continue
			input_text_embed_char_text_inp_tensors, _ = self.prepare_data(input_data=input_data,
																		  embed=embed,
																		  input_text_lang_tuple=input_text_lang_tuple,
																		  vocab_chars=vocab_chars,
																		  vocab_lang=vocab_lang)
			# transfer tensors to GPU if available
			if (self.system_parameters['cuda_is_avail']):
				input_text_embed_char_text_inp_tensors = input_text_embed_char_text_inp_tensors.cuda()
			n_highest_probs = 5
			self.evaluate_and_print(gru_model=gru_model, input_text_embed_char_text_inp_tensors=input_text_embed_char_text_inp_tensors,
									n_highest_probs=n_highest_probs, input_text=input_text, index2lang=index2lang)

	def str_to_int(self, string):
		try:
			number = int(string)
			return number
		except:
			print("Not a number")
			return 0

	def sample_tweets(self, tweet_retriever, vocab_lang):
		amount = self.str_to_int(input("Specify an amount and press Enter to sample live tweets: "))
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

	def retrieve_text(self, terminal_live_tweets, index2lang, tweet_retriever, vocab_lang):
		if terminal_live_tweets:
			sample_tweets = self.sample_tweets(tweet_retriever, vocab_lang)
			print('sample_tweets',sample_tweets)
			if sample_tweets is None:
				return None, None
			input_text =  list(sample_tweets.values())
			input_text_lang_tuple = [(text, index2lang[0]) for text in input_text]
		else:
			input_text = [input('Enter text: ')]
			input_text_lang_tuple = [(input_text[0], index2lang[0])]  # language must be in vocab_lang
		return input_text, input_text_lang_tuple

	def prepare_data(self, input_data, embed, input_text_lang_tuple, vocab_chars, vocab_lang):
		filtered_texts_and_lang = input_data.filter_out_irrelevant_tweet_parts(input_text_lang_tuple)
		# print('filtered_texts_and_lang',filtered_texts_and_lang)
		input_text_only_vocab_chars = input_data.get_texts_with_only_vocab_chars(filtered_texts_and_lang, vocab_chars)
		# print('input_text_only_vocab_chars', input_text_only_vocab_chars)
		input_text_indexed = input_data.get_indexed_texts_and_lang(input_text_only_vocab_chars, vocab_chars, vocab_lang)
		# print('input_text_indexed',input_text_indexed)
		input_text_embed_char_text_inp_tensors, input_text_target_tensors = input_data.create_embed_input_and_target_tensors \
			(indexed_texts_and_lang=input_text_indexed,
			 embed_weights_rel_path=self.system_parameters['trained_embed_weights_rel_path'],
			 embed=embed)
		return input_text_embed_char_text_inp_tensors, input_text_target_tensors

	def evaluate_and_print(self, gru_model, input_text_embed_char_text_inp_tensors, n_highest_probs, input_text, index2lang):
		evaluator = Evaluator.Evaluator(gru_model)
		for i, input_tensor in enumerate(input_text_embed_char_text_inp_tensors):
			lang_prediction, _ = evaluator.evaluate_single_date(input_tensor, n_highest_probs)
			print("====================\ntweet detected: \n\n%s\n" % input_text[i])

			# print n_highest_probs for input
			print('Language:')
			for i in range(len(lang_prediction)):
				if (i == 0):
					print("{0:.2f}".format(lang_prediction[i][0] * 100) + "%: " + str(
						index2lang[lang_prediction[i][1]]) + "\n")
				else:
					print(
						"{0:.2f}".format(lang_prediction[i][0] * 100) + "%: " + str(index2lang[lang_prediction[i][1]]))