import math
import numpy as np
from scipy import stats
import torch
from collections import Counter

class Evaluator(object):

    def __init__(self, model):
        self.model = model

    def evalute_data_set(self, input_data, target_data, vocab_lang, n_highest_probs=1):
        if(len(input_data) != len(target_data)):
            print("input and target size different for 'evaluate_data_set()'")
            return -1
        pred_true = 0
        predictions = []
        target_list = []
        val_loss = 0
        for input, target in zip(input_data, target_data):
            output,_ = self.model(input)
            val_loss += self.model.criterion(output, target)
            lang_prediction = self.evaluate_prediction(output, n_highest_probs)
#            print(lang_prediction)
            # checks if language prediction equals most common language in target (in case there are multiple targets)
            # todo later: multiple language predictions
            #print('lang_pred %s - target %s'%(lang_prediction, stats.mode(target.data.numpy()).mode[0]))
            target_list.append(stats.mode(target.data.numpy()).mode[0])
            predictions.append(lang_prediction[0][1])
            pred_true += int(lang_prediction[0][1] == target_list[-1])
#        print('val_loss', val_loss)
        accuracy = pred_true/len(input_data)
#        self.confusion_matrix(predictions, target_list, vocab_lang)
#        print('accuracy', accuracy)
        return accuracy, lang_prediction

#todo
    def evaluate_single_date(self, input, target, vocab_lang, n_highest_probs):
        pass

    #prediction: tensor of languages-dimensional entries containing log softmax probabilities
    def evaluate_prediction(self, prediction, n_highest_probs):
        lang_predictions = np.zeros([prediction.size()[1]])
        pred_size = prediction.size()[0]
        for pred in prediction:
            for i in range(len(lang_predictions)):
                lang_predictions[i] += np.exp(pred[i].data[0])
        lang_predictions = [lang_mean / pred_size for lang_mean in lang_predictions]
        languages_probs_and_idx = [(lang_predictions[i], i) for i in range(len(lang_predictions))]
        languages_probs_and_idx.sort(reverse=True)
        highest_probs = [languages_probs_and_idx[i] for i in range(min(n_highest_probs, len(languages_probs_and_idx)))]
        return highest_probs

    def confusion_matrix(self, predictions, targets, vocab_lang):
        #todo: more elaborate
        if(len(predictions) != len(targets)):
            print("predictions and target size different for 'confusion_matrix()'")
            return []
        conf_matrix = np.zeros((len(vocab_lang), len(vocab_lang)))
        for pred, targ in zip(predictions, targets):
            conf_matrix[targ][pred] += 1
        print('conf_matrix\n', conf_matrix)
