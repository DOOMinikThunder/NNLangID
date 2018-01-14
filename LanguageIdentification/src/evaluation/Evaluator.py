import math
import numpy as np
from scipy import stats
import torch
from collections import Counter

class Evaluator(object):

    def __init__(self, model):
        self.model = model

    def evaluate_data_set(self, input_data, target_data, vocab_lang, n_highest_probs=1):
        """
        evaluates a whole tweet_retriever_data set
        :param input_data: input tweet_retriever_data set
        :param target_data: target set of same size as input tweet_retriever_data
        :param vocab_lang: dictionary containing 'language':(index, frequency)
        :param n_highest_probs: the n languages with the highest probabilites to be calculated
        :return: accuracy, confusion matrix
        """
        if(len(input_data) != len(target_data)):
            print("input and target size different for 'evaluate_data_set()'")
            return -1
        pred_true = 0
        predictions = []
        target_list = []
        for input, target in zip(input_data, target_data):
            lang_prediction = self.evaluate_single_date(input, n_highest_probs)
            target_list.append(stats.mode(target.data.numpy()).mode[0])
            predictions.append(lang_prediction[0][1])
            pred_true += int(lang_prediction[0][1] == target_list[-1])
        accuracy = pred_true/len(input_data)
        conf_matrix = self.confusion_matrix(predictions, target_list, vocab_lang)
        return accuracy, conf_matrix

    def evaluate_single_date(self, input, n_highest_probs):
        """
        evaluates a single date
        :param input: the input date/tweet
        :param n_highest_probs: the n languages with the highest probabilites to be calculated
        :return: list of (probability, language index)
        """
        hidden = self.model.initHidden()
        output,hidden = self.model(input, hidden)
        lang_prediction = self.evaluate_prediction(output, n_highest_probs)
        return lang_prediction

    #prediction: tensor of languages-dimensional entries containing log softmax probabilities
    def evaluate_prediction(self, prediction, n_highest_probs):
        """
        given a single date (tweet), calculates the respective language predictions
        :param prediction: the predicted date/tweet
        :param n_highest_probs: the n languages with the highest probabilites
        :return: list of (probability, language index)
        """
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
        """
        row = target language, column = predicted language
        :param predictions: single language predictions
        :param targets: single language targets
        :param vocab_lang: dict containing all languages
        :return: confusion matrix
        """
        if(len(predictions) != len(targets)):
            print("predictions and target size different for 'confusion_matrix()'")
            return []
        conf_matrix = np.zeros((len(vocab_lang), len(vocab_lang)))
        for pred, targ in zip(predictions, targets):
            conf_matrix[targ][pred] += 1
        return conf_matrix

    def to_string_confusion_matrix(self, confusion_matrix, vocab_lang, pad):
        """
        converts calculated confusion matrix to string
        :param confusion_matrix: the confusion matrix
        :param vocab_lang: dict containing 'language':(index, frequency)
        :param pad: padding for printout, use a number >= len(max(frequency in vocab lang)
        :return: string confusion matrix
        """
        idx_lang = []
        for language, (language_idx, _)  in vocab_lang.items():
            idx_lang.append((language_idx, language))
        idx_lang.sort()
        horizontal_lang = ""
        for _, lang in idx_lang:
            horizontal_lang += '{0: >{pad}}'.format(lang, pad=pad)+"\t"
        print_matrix = "t\p\t" + horizontal_lang +"\taccuracy\n"
        for i,row in enumerate(confusion_matrix):
            row_accuracy = (row[i]/sum(row))*100
            print_matrix += idx_lang[i][1] + "\t" + self.row_as_string(row, pad) + '{0: >9}'.format(str(row_accuracy)) + "%\n"
        return print_matrix

    def row_as_string(self, row, pad):
        """
        single confusion matrix row as a string
        :param row: the confusion matrix row
        :param pad: padding for printout, use a number >= len(max(frequency in vocab lang)
        :return: the string row
        """
        str_row = ""
        for item in row:
            str_row += '{0: >{pad}}'.format(str(int(item)), pad=pad) + "\t"
        return str_row
