from input import InputData
from net import GRUModel
from evaluation import Evaluator

class UseModel(object):
	def __init__(self, system_parameters):
		self.system_parameters = system_parameters


	def train(self, data_sets, vocab_lang, vocab_chars, model_save_path):
		gru_model, input_and_target_tensors = self.load_model_and_data(data_sets, self.system_parameters['trained_embed_weights_rel_path'])
		if(len(data_sets)<2):
			print("ERROR: two data sets (traninig, validation) are needed")
			return
		print('Model:\n', gru_model)
		self.loop_training(gru_model, input_and_target_tensors, vocab_lang, vocab_chars, model_save_path)

	def evaluate_validation(self, epoch, evaluator, val_input, val_target, vocab_lang):
		val_mean_loss, predictions, targets = evaluator.evaluate_data_set(val_input,
																		  val_target,
																		  vocab_lang)
		cur_val_accuracy = evaluator.accuracy(predictions, targets)
		val_conf_matrix = evaluator.confusion_matrix(predictions, targets, vocab_lang)

		to_print = [('confusion_matrix\n: ',
					 evaluator.to_string_confusion_matrix(confusion_matrix=val_conf_matrix, vocab_lang=vocab_lang,
														  pad=5)),
					('Epoch: ', epoch),
					('validation set accuracy: ', cur_val_accuracy),
					('validation mean loss: ', val_mean_loss)]
		self.print(to_print)
		return val_mean_loss

	def test(self, data_sets, model_path):
		gru_model, input_and_target_tensors = self.load_model_and_data(data_sets, self.system_parameters['embed_weights_rel_path'])

		start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(
			model_path)

		evaluator = Evaluator.Evaluator(gru_model)

		mean_loss, accuracy, confusion_matrix, precision, recall, f1_score = evaluator.all_metrics(input_and_target_tensors[0][0],
																								   input_and_target_tensors[0][1],
																								   vocab_lang)
		to_print = [('confusion matrix: \n', evaluator.to_string_confusion_matrix(confusion_matrix, vocab_lang, 5)),
					('precision: ', precision),
					('recall: ', recall),
					('f1 score: ', f1_score),
					('Epochs trained: ', start_epoch),
					('Best validation set accuracy: ', best_val_accuracy),
					('Test set accuracy: ', test_accuracy),
					('System parameters used:: ', self.system_parameters)]
		self.print(to_print)

		# save test_accuracy to file
		gru_model.save_model_checkpoint_to_file({
			'start_epoch': start_epoch,
			'best_val_accuracy': best_val_accuracy,
			'test_accuracy': test_accuracy,
			'state_dict': gru_model.state_dict(),
			'optimizer': gru_model.optimizer.state_dict(),
			'system_param_dict': system_param_dict,
			'vocab_chars': vocab_chars,
			'vocab_lang': vocab_lang,
		},
			model_path)

	def prepare_data(self, data_sets, embed_weights_path):
		input_data = InputData.InputData()
		embed, num_classes = input_data.create_embed_from_weights_file(embed_weights_path)
		input_and_target_tensors = []
		for data_set in data_sets:
			inp, target = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=data_set,
																			embed_weights_rel_path=self.system_parameters['embed_weights_rel_path'],
																			embed=embed)
			input_and_target_tensors.append((inp, target))
		return input_and_target_tensors, embed, num_classes

	def load_model(self, embed, num_classes):
		return GRUModel.GRUModel(input_size=embed.weight.size()[1],  # equals embedding dimension
									  hidden_size=self.system_parameters['hidden_size_rnn'],
									  num_layers=self.system_parameters['num_layers_rnn'],
									  num_classes=num_classes,
									  is_bidirectional=self.system_parameters['is_bidirectional'],
									  initial_lr=self.system_parameters['initial_lr_rnn'],
									  weight_decay=self.system_parameters['weight_decay_rnn'])

	def load_model_and_data(self, data_sets, embed_weights_path):
		input_and_target_tensors, embed, num_classes = self.prepare_data(data_sets=data_sets, embed_weights_path=embed_weights_path)

		gru_model = self.load_model(embed=embed, num_classes=num_classes)
		# run on GPU if available
		if (self.system_parameters['cuda_is_avail']):
			for i in range(len(input_and_target_tensors)):
				input_and_target_tensors[i][0] = [tensor.cuda() for tensor in input_and_target_tensors[i][0]]
				input_and_target_tensors[i][1] = [tensor.cuda() for tensor in input_and_target_tensors[i][1]]
			gru_model.cuda()
		return gru_model, input_and_target_tensors



	def loop_training(self, gru_model, input_and_target_tensors, vocab_lang, vocab_chars, model_save_path):
		cur_val_accuracy = 0
		best_val_accuracy = -1.0
		epoch = 0
		is_improving = True
		evaluator = Evaluator.Evaluator(gru_model)

		# stop training when validation set error stops getting smaller ==> stop when overfitting occurs
		# or when maximum number of epochs reached
		while is_improving and not epoch == self.system_parameters['num_epochs_rnn']:
			print('RNN epoch:', epoch)
			# inputs: whole data set, every date contains the embedding of one char in one dimension
			# targets: whole target set, target is set for each character embedding
			gru_model.train(inputs=input_and_target_tensors[0][0],
							targets=input_and_target_tensors[0][1],
							batch_size=self.system_parameters['batch_size_rnn'])
			# evaluate validation set
			val_mean_loss = self.evaluate_validation(epoch, evaluator, input_and_target_tensors[1][0], input_and_target_tensors[1][1], vocab_lang)

			# check if accuracy improved and if so, save model checkpoint to file
			if (best_val_accuracy < cur_val_accuracy):
				best_val_accuracy = cur_val_accuracy
				gru_model.save_model_checkpoint_to_file({
					'start_epoch': epoch + 1,
					'best_val_accuracy': best_val_accuracy,
					'test_accuracy': -1.0,
					'state_dict': gru_model.state_dict(),
					'optimizer': gru_model.optimizer.state_dict(),
					'system_param_dict': self.system_parameters,
					'vocab_chars': vocab_chars,
					'vocab_lang': vocab_lang,
				},
					model_save_path)
			else:
				is_improving = False
			epoch += 1



	def print(self, string_date_tuple):
		for string, date in string_date_tuple:
			print('========================================')
			print(string, date, sep="")
		print('========================================')
