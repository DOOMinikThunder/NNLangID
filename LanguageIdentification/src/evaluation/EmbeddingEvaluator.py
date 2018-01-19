import math
import numpy as np
from scipy import stats
import torch
from torch.autograd import Variable
from collections import Counter
from evaluation import Evaluator

class EmbeddingEvaluator(object):

    def __init__(self, model):
        """

        Args:
        	model:
        """
        self.model = model
#        super(EmbeddingEvaluator, self).__init__(model)
    
    def evaluate_data_set(self, val_batched_pairs, num_neg_samples):
        """

        Args:
        	val_batched_pairs:
        	num_neg_samples:

        Returns:

        """
        for batch_i, batch in enumerate(val_batched_pairs):
            batch_size = len(batch)
            
            targets_1_pos = [pair[0] for pair in batch]
            contexts_1_pos = [pair[1] for pair in batch]
            contexts_0_pos_samples = self.model.get_neg_samples(batch_size, num_neg_samples)

            targets_1_pos = Variable(torch.LongTensor(targets_1_pos))
            contexts_1_pos = Variable(torch.LongTensor(contexts_1_pos))
            contexts_0_pos_samples = Variable(torch.LongTensor(contexts_0_pos_samples))
            # transfer tensors to GPU if available
            if (self.model.cuda_is_avail):
                targets_1_pos = targets_1_pos.cuda()
                contexts_1_pos = contexts_1_pos.cuda()
                contexts_0_pos_samples = contexts_0_pos_samples.cuda()         

            batch_mean_loss = self.model.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
        return batch_mean_loss.data[0]