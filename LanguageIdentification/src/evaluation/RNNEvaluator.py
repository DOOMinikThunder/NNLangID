from __future__ import division
import math
import numpy as np
from scipy import stats
import torch

#import Evaluator

class RNNEvaluator(object):

    def __init__(self, model):
        self.model = model
        self.cuda_is_avail = torch.cuda.is_available()
#        super(RNNEvaluator, self).__init__(model)

    def all_metrics(self, input_data, target_data, vocab_lang):
        """
        evaluates a data set and returns all possible metrics
        Args:
        	input_data: tweets to evaluate
        	target_data: target language of tweets
        	vocab_lang: dict containing all languages and their indices

        Returns:
        	mean_loss: average loss of evaluation
        	accuracy: overall accuracy
        	confusion matrix: detailed matrix of how each tweet was evaluated
        	precision: list of precision for each language
        	recall: list of recall for each language
        	f1_score: list of f1_score for each language

        """
        mean_loss, predictions, targets = self.evaluate_data_set_basic(input_data,
                                                                       target_data,
                                                                       n_highest_probs=1)
        accuracy = self.accuracy(predictions, targets)
        confusion_matrix = self.confusion_matrix(predictions, targets, vocab_lang)
        precision = self.precision(confusion_matrix)
        recall = self.recall(confusion_matrix)
        f1_score = self.f1_score(precision, recall)
        return mean_loss, accuracy, confusion_matrix, precision, recall, f1_score

    def evaluate_data_set(self, input_data, target_data, n_highest_probs=1):
        """
        evaluates a data set as fast as possible
        by merging needed functions and elminating if clauses
        Args:
        	input_data: tweets to evaluate
        	target_data: target languages of each char in each tweet
        	n_highest_probs: n highest probability predictions for each tweet

        Returns:
        	mean_loss: average loss over all tweets
        	accuracy: overall accuracy

        """
        predictions = []
        target_list = []
        acc_loss = []
        for input, target in zip(input_data, target_data):
            hidden = self.model.initHidden()
            output, hidden = self.model(input, hidden)
            loss = self.model.criterion(output, target)
            lang_predictions = np.zeros([output.size()[1]])
            pred_size = output.size()[0]
            for pred in output:
                for i in range(len(lang_predictions)):
                    lang_predictions[i] += np.exp(pred[i].data[0])
            lang_predictions = [lang_mean / pred_size for lang_mean in lang_predictions]
            languages_probs_and_idx = [(lang_predictions[i], i) for i in range(len(lang_predictions))]
            languages_probs_and_idx.sort(reverse=True)
            lang_prediction = [languages_probs_and_idx[i] for i in
                             range(min(n_highest_probs, len(languages_probs_and_idx)))]

            acc_loss.append(loss)
            target_list.append(target.data[0])
            predictions.append(lang_prediction[0][1])
        mean_loss = sum(acc_loss) / float(len(acc_loss))
        pred_true = 0
        for pred, target in zip(predictions, target_list):
            pred_true += int(pred == target)
        accuracy = pred_true / len(target_list)
        return mean_loss.data[0], accuracy

    def evaluate_data_set_basic(self, input_data, target_data, n_highest_probs=1):
        """
        evaluates a data set
        Args:
        	input_data: tweets to evaluate
        	target_data: target languages of each char in each tweet
        	n_highest_probs: n highest probability predictions for each tweet

        Returns:
        	mean_loss: average loss over all tweets
        	predictions: predictions for each tweet
        	target_list: target for each tweet

        """
        if (len(input_data) != len(target_data)):
            print("input and target size different for 'evaluate_data_set_basic()'")
            return -1
        predictions = []
        target_list = []
        acc_loss = []
        for input, target in zip(input_data, target_data):
            lang_prediction, loss = self.evaluate_single_date(input, n_highest_probs, target)
            acc_loss.append(loss)
            # transfer back from GPU to CPU if GPU available
            target_list.append(target.data[0])
            predictions.append(lang_prediction[0][1])
        mean_loss = sum(acc_loss) / float(len(acc_loss))
        return mean_loss.data[0], predictions, target_list

    def accuracy(self, predictions, targets):
        """
        computes the accuracy over all prediction-target pairs
        Args:
        	predictions: predictions for each tweet
        	target_list: target for each tweet

        Returns:
            accuracy: overall accuracy
        """
        pred_true = 0
        for pred, target in zip(predictions, targets):
            pred_true += int(pred == target)
        return pred_true / len(targets)

    def f1_score(self, precision, recall):
        """
        f1 score is the harmonic mean over precision and recall
        Args:
        	precision: list of precision for each language
        	recall: list of recall for each language

        Returns:
        	f1_score: list of f1_score for each language

        """
        try:
            return [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
        except ZeroDivisionError:
            return [0]

    def recall(self, confusion_matrix):
        """
        recall is a measurement to tell if a language was often predicted although it should not have been
        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated

        Returns:
            recall: list of recall for each language
        """
        true_positives = self.true_positives(confusion_matrix)
        false_negatives = self.false_negatives(confusion_matrix)
        try:
            return [tp / (tp + sum(fn)) for tp, fn in zip(true_positives, false_negatives)]
        except ZeroDivisionError:
            return [0]

    def precision(self, confusion_matrix):
        """
        precision is a measurement to tell if a language was often classified as another one
        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated

        Returns:
            precision: list of precision for each language
        """
        true_positives = self.true_positives(confusion_matrix)
        false_positives = self.false_positive(confusion_matrix)
        try:
            return [tp / (tp + sum(fp)) for tp, fp in zip(true_positives, false_positives)]
        except ZeroDivisionError:
            return [0]

    def __true_positives(self, confusion_matrix):
        """
        true positives equals the trace of the confusion matrix
        i.e. prediction = target
        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated

        Returns:
        	true_positives: list of all true_positives for each language

        """
        return [confusion_matrix[i][i] for i in range(len(confusion_matrix))]

    def __false_positive(self, confusion_matrix):
        """
        false positive equals the sum of one row of the confusion matrix
        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated

        Returns:
        	false_positives: list of all false_positives for each language(row)

        """
        return [confusion_matrix[i][:i] + confusion_matrix[i][i + 1:] for i in range(len(confusion_matrix))]

    def __false_negatives(self, confusion_matrix):
        """
        false negative equals the sum one column of the confusion matrix
        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated

        Returns:
        	false_negatives: list of all false_negatives for each language(column)

        """
        matrix_in_columns = list(zip(*confusion_matrix))
        return [matrix_in_columns[i][:i] + matrix_in_columns[i][i + 1:] for i in range(len(matrix_in_columns))]

    def evaluate_single_date(self, input, n_highest_probs, target=None):
        """
        evaluates one tweet
        Args:
        	input: input tweet
        	n_highest_probs: n highest probabilities of language predictions
        	target: tweet's target

        Returns:
        	lang_predictions: list of highest probability-language pairs for n languages
        	loss: loss of prediction to target

        """
        hidden = self.model.initHidden()
        output, hidden = self.model(input, hidden)
        if target is not None:
            loss = self.model.criterion(output, target)
        else:
            loss = 0
        lang_prediction = self.evaluate_prediction(output, n_highest_probs)
        return lang_prediction, loss

    def evaluate_prediction(self, prediction, n_highest_probs):
        """
        Given a prediction, computes the most likely languages
        Args:
        	prediction: tensor of languages-dimensional entries containing log softmax probabilities
        	n_highest_probs: n highest probabilities of languages to return

        Returns:
        	highest_probs: list of highest probability-language pairs for n languages

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
        Confusion matrix has language x language entries, corresponding to predictions(row) and targets(column)
        Args:
        	predictions: predicted languages
        	targets: target languages
        	vocab_lang: dict containing languages and there unique indices

        Returns:
        	conf_matrix: detailed matrix of how each tweet was evaluated

        """
        if (len(predictions) != len(targets)):
            print("predictions and target size different for 'confusion_matrix()'")
            return []
        conf_matrix = np.zeros((len(vocab_lang), len(vocab_lang)))
        for pred, targ in zip(predictions, targets):
            conf_matrix[targ][pred] += 1
        return conf_matrix.tolist()

    def to_string_confusion_matrix(self, confusion_matrix, vocab_lang, pad):
        """

        Args:
        	confusion_matrix: detailed matrix of how each tweet was evaluated
        	vocab_lang: dict containing languages and there unique indices
        	pad: padding to insert to keep the string readable

        Returns:
        	print_matrix: string conversion of confustion_matrix

        """
        idx_lang = []
        for language, (language_idx, _) in vocab_lang.items():
            idx_lang.append((language_idx, language))
        idx_lang.sort()
        horizontal_lang = ""
        for _, lang in idx_lang:
            horizontal_lang += '{0: >{pad}}'.format(lang, pad=pad) + "\t"
        print_matrix = "t\p\t" + horizontal_lang + "\taccuracy\n"
        for i, row in enumerate(confusion_matrix):
            row_accuracy = (row[i] / sum(row)) * 100
            print_matrix += idx_lang[i][1] + "\t" + self.row_as_string(row, pad) + '{0: >9.3}'.format(
                str(row_accuracy)) + "%\n"
        return print_matrix

    def row_as_string(self, row, pad):
        """

        Args:
        	row: one row of a confusion_matrix
        	pad: padding to insert to keep the string readable

        Returns:
        	str_row: the row as string

        """
        str_row = ""
        for item in row:
            str_row += '{0: >{pad}}'.format(str(int(item)), pad=pad) + "\t"
        return str_row