from input import InputData
from net import GRUModel
#from evaluation import RNNEvaluator



class RNNCalculation(object):
    
    
    def __init__(self, system_parameters):
        self.system_parameters = system_parameters


    def train_rnn(self, data_sets, vocab_chars, vocab_lang, model_save_path):
        input_and_target_tensors, gru_model = self.get_model_and_data(data_sets)
        if (len(data_sets) < 2):
            print("ERROR: Two data sets (traninig, validation) are needed!")
            return
        print('Model:\n', gru_model)
        self.loop_training(gru_model, input_and_target_tensors, vocab_lang, vocab_chars, model_save_path)


    def evaluate_validation(self, epoch, evaluator, val_input, val_target, vocab_lang):
        val_mean_loss, predictions, targets = evaluator.evaluate_data_set(val_input,
                                                                          val_target,
                                                                          vocab_lang)
        cur_val_accuracy = evaluator.accuracy(predictions, targets)
        val_conf_matrix = evaluator.confusion_matrix(predictions, targets, vocab_lang)

        to_print = [('confusion_matrix: \n',
                     evaluator.to_string_confusion_matrix(confusion_matrix=val_conf_matrix, vocab_lang=vocab_lang,
                                                          pad=5)),
                    ('Epoch: ', epoch),
                    ('validation set accuracy: ', cur_val_accuracy),
                    ('validation mean loss: ', val_mean_loss)]
        self.print_out(to_print)
        return val_mean_loss


    def test_rnn(self, data_sets, model_path):
        input_and_target_tensors, gru_model = self.get_model_and_data(data_sets)

        start_epoch, best_val_accuracy, test_accuracy, system_param_dict, vocab_chars, vocab_lang = gru_model.load_model_checkpoint_from_file(model_path)

        evaluator = RNNEvaluator.RNNEvaluator(gru_model)

        mean_loss, test_accuracy, confusion_matrix, precision, recall, f1_score = evaluator.all_metrics(input_and_target_tensors[0][0],
                                                                                                                               input_and_target_tensors[0][1],
                                                                                                                              vocab_lang)
        to_print = [('confusion matrix: \n', evaluator.to_string_confusion_matrix(confusion_matrix, vocab_lang, 5)),
                    ('precision: ', precision),
                    ('recall: ', recall),
                    ('f1 score: ', f1_score),
                    ('Epochs trained: ', start_epoch),
                    ('Best validation set accuracy: ', best_val_accuracy),
                    ('Test set accuracy: ', test_accuracy),
                    ('System parameters used: ', self.system_parameters)]
        self.print_out(to_print)

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


    def get_model_and_data(self, data_sets):
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(self.system_parameters['embed_weights_rel_path'])
        input_and_target_tensors = []
        for data_set in data_sets:
            inputs, targets = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=data_set,
                                                                               embed_weights_rel_path=self.system_parameters['embed_weights_rel_path'],
                                                                               embed=embed)
            # transfer tensors to GPU if available
            if (self.system_parameters['cuda_is_avail']):
                input_and_target_tensors.append(([tensor.cuda() for tensor in inputs], [tensor.cuda() for tensor in targets]))
            else:
                input_and_target_tensors.append((inputs, targets))

        gru_model = GRUModel.GRUModel(input_size=embed.weight.size()[1],  # equals embedding dimension
                                      hidden_size=self.system_parameters['hidden_size_rnn'],
                                      num_layers=self.system_parameters['num_layers_rnn'],
                                      num_classes=num_classes,
                                      is_bidirectional=self.system_parameters['is_bidirectional'],
                                      initial_lr=self.system_parameters['initial_lr_rnn'],
                                      weight_decay=self.system_parameters['weight_decay_rnn'])
        # run on GPU if available
        if (self.system_parameters['cuda_is_avail']):
            gru_model.cuda()
        return input_and_target_tensors, gru_model


    def loop_training(self, gru_model, input_and_target_tensors, vocab_lang, vocab_chars, model_save_path):
#        cur_val_accuracy = 0
#        best_val_accuracy = -1.0
#        best_val_mean_loss = float('inf')
#        cur_val_mean_loss = float('inf')
#        epoch = 0
#        is_improving = True
#        evaluator = RNNEvaluator.RNNEvaluator(gru_model)
#
#        # stop training when validation set error stops getting smaller ==> stop when overfitting occurs
#        # or when maximum number of epochs reached
#        while is_improving and not epoch == self.system_parameters['num_epochs_rnn']:
#            print('RNN epoch:', epoch)
        
        
        
        # inputs: whole data set, every date contains the embedding of one char in one dimension
        # targets: whole target set, target is set for each character embedding
        gru_model.train(train_inputs=input_and_target_tensors[0][0],
                        train_targets=input_and_target_tensors[0][1],
                        val_inputs=input_and_target_tensors[1][0],
                        val_targets=input_and_target_tensors[1][1],
                        batch_size=self.system_parameters['batch_size_rnn'],
                        max_eval_checks_not_improved=self.system_parameters['max_eval_checks_not_improved_rnn'],
                        max_num_epochs=self.system_parameters['max_num_epochs_rnn'],
                        eval_every_num_batches=self.system_parameters['eval_every_num_batches_rnn'],
                        lr_decay_every_num_batches=self.system_parameters['lr_decay_every_num_batches_rnn'],
                        lr_decay_factor=self.system_parameters['lr_decay_factor_rnn'],
                        rnn_model_checkpoint_rel_path=self.system_parameters['rnn_model_checkpoint_rel_path'],
                        system_param_dict=self.system_parameters)
            
            
#            
#            # evaluate validation set
##            cur_val_mean_loss = self.evaluate_validation(epoch, evaluator, input_and_target_tensors[1][0], input_and_target_tensors[1][1], vocab_lang)
#
#            cur_val_mean_loss, best_val_accuracy, confusion_matrix, precision, recall, f1_score = evaluator.all_metrics(input_and_target_tensors[1][0],
#                                                                                                                input_and_target_tensors[1][1],
#                                                                                                                vocab_lang)
#            # check if accuracy improved and if so, save model checkpoint to file
#            if (best_val_mean_loss > cur_val_mean_loss):
#                best_val_mean_loss = cur_val_mean_loss
#                print('best_val_mean_loss', best_val_mean_loss)
#                gru_model.save_model_checkpoint_to_file({
#                    'start_epoch': epoch + 1,
#                    'best_val_accuracy': best_val_accuracy,
#                    'test_accuracy': -1.0,
#                    'state_dict': gru_model.state_dict(),
#                    'optimizer': gru_model.optimizer.state_dict(),
#                    'system_param_dict': self.system_parameters,
#                    'vocab_chars': vocab_chars,
#                    'vocab_lang': vocab_lang,
#                },
#                    model_save_path)
#            else:
#                is_improving = False
#            epoch += 1





    def print_out(self, string_date_tuple):
        for string, date in string_date_tuple:
            print('========================================')
            print(string, date)
        print('========================================')
