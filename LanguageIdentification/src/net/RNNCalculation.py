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


import pprint
from input import InputData
from embedding import SkipGramModel
from net import GRUModel
from evaluation import RNNEvaluator



class RNNCalculation(object):
    
    
    def __init__(self, system_param_dict):
        """

        Args:
        	system_param_dict:
        """
        self.system_param_dict = system_param_dict


    def train_rnn(self, data_sets, vocab_chars, vocab_lang):
        """

        Args:
        	data_sets:
        	vocab_chars:
        	vocab_lang:

        Returns:

        """
        input_and_target_tensors, gru_model = self.get_model_and_data(data_sets, vocab_chars, vocab_lang, self.system_param_dict['embed_weights_rel_path'], True)
        if (len(data_sets) < 2):
            print("ERROR: Two data sets (training, validation) are needed!")
            return
        print('Model:\n', gru_model)
        
        # inputs: whole data set, every date contains the embedding of one char in one dimension
        # targets: whole target set, target is set for each character embedding
        gru_model.train(train_inputs=input_and_target_tensors[0][0],
                        train_targets=input_and_target_tensors[0][1],
                        val_inputs=input_and_target_tensors[1][0],
                        val_targets=input_and_target_tensors[1][1])


    def test_rnn(self, data_sets, vocab_chars, vocab_lang):
        """

        Args:
        	data_sets:
        	vocab_chars:
        	vocab_lang:

        Returns:

        """
        input_and_target_tensors, gru_model = self.get_model_and_data(data_sets, vocab_chars, vocab_lang, self.system_param_dict['embed_weights_rel_path'], True)
        state = gru_model.load_model_checkpoint_from_file(self.system_param_dict['rnn_model_checkpoint_rel_path'])
        results_dict = state['results_dict']
        rnn_evaluator = RNNEvaluator.RNNEvaluator(gru_model)

        test_mean_loss, test_accuracy, confusion_matrix, precision, recall, f1_score = rnn_evaluator.all_metrics(input_and_target_tensors[0][0],
                                                                                                                 input_and_target_tensors[0][1],
                                                                                                                 vocab_lang)
        confusion_matrix = rnn_evaluator.to_string_confusion_matrix(confusion_matrix, vocab_lang, 5)
        
        print('Test results:')
        to_print = [('Confusion matrix:\n', confusion_matrix),
                    ('Best validation set mean loss:', results_dict['best_val_mean_loss']),
                    ('Test set mean loss:', test_mean_loss),
                    ('Best validation set accuracy:', results_dict['best_val_accuracy']),
                    ('Test set accuracy:', test_accuracy),
                    ('Precision:', precision),
                    ('Recall:', recall),
                    ('F1 score:', f1_score),
                    ('Epochs trained:', results_dict['start_epoch']),
                    ('Total batches trained:', results_dict['start_total_trained_batches_counter']),
                    ('System parameters used:', self.system_param_dict)]
        self.print_out(to_print)

        # save with new results to file
        results_dict['test_mean_loss'] = test_mean_loss
        results_dict['test_accuracy'] = test_accuracy
        results_dict['confusion_matrix'] = confusion_matrix
        results_dict['precision'] = precision
        results_dict['recall'] = recall
        results_dict['f1_score'] = f1_score
        state['results_dict'] = results_dict
        gru_model.save_model_checkpoint_to_file(state, self.system_param_dict['rnn_model_checkpoint_rel_path'])


    def print_model_checkpoint(self, vocab_chars, vocab_lang, is_rnn_model):
        """

        Args:
        	vocab_chars:
        	vocab_lang:

        Returns:

        """
        _, model = self.get_model_and_data([], vocab_chars, vocab_lang, self.system_param_dict['print_model_checkpoint_embed_weights'], is_rnn_model)
        # check which model shall be printed
        if (is_rnn_model):
            model_checkpoint = self.system_param_dict['print_rnn_model_checkpoint']
        else:
            model_checkpoint = self.system_param_dict['print_embed_model_checkpoint']
        state = model.load_model_checkpoint_from_file(model_checkpoint)
        system_param_dict = state['system_param_dict']
        results_dict = state['results_dict']
        rnn_evaluator = RNNEvaluator.RNNEvaluator(model)
        
        conf_matrix_exists = False
        # get nice formatting for confusion matrix
        if ('confusion_matrix' in results_dict):
            conf_matrix_exists = True
            confusion_matrix = rnn_evaluator.to_string_confusion_matrix(results_dict['confusion_matrix'], vocab_lang, 5)
        
        results_print_dict = {}
        for result in results_dict:
            # only print relevant and not too verbose data
            if (result != 'state_dict'
                and result != 'optimizer'
                and result != 'vocab_chars'
                and result != 'vocab_lang'
                and result != 'confusion_matrix'):
                results_print_dict[result] = results_dict[result]
        
        print('Model checkpoint data:')
        to_print = [('Model:\n', model),
                    ('System parameters:', system_param_dict),
                    ('Results:', results_print_dict)]
        if (conf_matrix_exists):
            to_print.append(('Confusion matrix:\n', confusion_matrix))
        self.print_out(to_print)
        

    def get_model_and_data(self, data_sets, vocab_chars, vocab_lang, embed_weights_rel_path, is_rnn_model):
        """

        Args:
        	data_sets:
        	vocab_chars:
        	vocab_lang:
        	embed_weights_rel_path:

        Returns:

        """
        input_data = InputData.InputData()
        embed, num_classes = input_data.create_embed_from_weights_file(embed_weights_rel_path)
        input_and_target_tensors = []
        for data_set in data_sets:
            inputs, targets = input_data.create_embed_input_and_target_tensors(indexed_texts_and_lang=data_set,
                                                                               embed_weights_rel_path=embed_weights_rel_path,
                                                                               embed=embed)
            # transfer tensors to GPU if available
            if (self.system_param_dict['cuda_is_avail']):
                input_and_target_tensors.append(([tensor.cuda() for tensor in inputs], [tensor.cuda() for tensor in targets]))
            else:
                input_and_target_tensors.append((inputs, targets))

        # choose which model to create
        if (is_rnn_model):
            model = GRUModel.GRUModel(vocab_chars=vocab_chars,
                                      vocab_lang=vocab_lang,
                                      input_size=embed.weight.size()[1],  # equals embedding dimension
                                      num_classes=num_classes,
                                      system_param_dict=self.system_param_dict)
        else:
            model = SkipGramModel.SkipGramModel(vocab_chars=vocab_chars,
                                                vocab_lang=vocab_lang,
                                                embed_dim=embed.weight.size()[1],
                                                system_param_dict=self.system_param_dict)
        # run on GPU if available
        if (self.system_param_dict['cuda_is_avail']):
            model.cuda()
        return input_and_target_tensors, model


    def print_out(self, string_date_tuple):
        """

        Args:
        	string_date_tuple:

        Returns:

        """
        for string, date in string_date_tuple:
            print('========================================')
            # print dicts with new line for every key
            if isinstance(date, dict):
                print(string)
                pprint.pprint(date)
            else:
                print(string, date)
        print('========================================')