import math
import numpy as np
from scipy import stats
import torch

class Evaluator(object):

    def __init__(self, model):
        self.model = model

    def evalute_data_set(self, input_data, target_data, vocab_lang):
        if(len(input_data) != len(target_data)):
            print("input and target size different for 'evaluate_data_set()'")
            return -1
        pred_true = 0
        predictions = []
        target_list = []
        for input, target in zip(input_data, target_data):
            output,_ = self.model(input)
            lang_prediction = self.evaluate_prediction(output)
            # checks if language prediction equals most common language in target (in case there are multiple targets)
            # todo later: multiple language predictions
            #print('lang_pred %s - target %s'%(lang_prediction, stats.mode(target.data.numpy()).mode[0]))
            target_list.append(stats.mode(target.data.numpy()).mode[0])
            predictions.append(lang_prediction)
            pred_true += int(lang_prediction == target_list[-1])
        
        accuracy = pred_true/len(input_data)
        self.confusion_matrix(predictions, target_list, vocab_lang)
#        print('accuracy', accuracy)
        return accuracy

    #prediction: tensor of languages-dimensional entries containing log softmax probabilities
    def evaluate_prediction(self, prediction):
        lang_predictions = np.zeros([prediction.size()[1]])
        pred_size = prediction.size()[0]
        for pred in prediction:
            for i in range(len(lang_predictions)):
                lang_predictions[i] += np.exp(pred[i].data[0])
        lang_predictions = [lang_mean / pred_size for lang_mean in lang_predictions]
        lang_idx = lang_predictions.index(max(lang_predictions))
        return lang_idx

    def confusion_matrix(self, predictions, targets, vocab_lang):
        #todo: more elaborate
        if(len(predictions) != len(targets)):
            print("predictions and target size different for 'confusion_matrix()'")
            return []
        conf_matrix = np.zeros((len(vocab_lang), len(vocab_lang)))
        for pred, targ in zip(predictions, targets):
            conf_matrix[targ][pred] += 1
        print('conf_matrix\n', conf_matrix)
