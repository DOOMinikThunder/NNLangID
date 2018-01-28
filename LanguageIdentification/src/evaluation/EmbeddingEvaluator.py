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


import torch
from torch.autograd import Variable


class EmbeddingEvaluator(object):
    """Class for embedding evaluation.
    """
    
    def __init__(self, model):
        """
        Args:
            model: The model to be evaluated.
        """
        self.model = model
    
    def evaluate_data_set(self, val_batched_pairs, num_neg_samples):
        """
        Evaluate the model on a given data set.
        
        Args:
            val_batched_pairs: List of validation batched pairs.
            num_neg_samples: Number of negative samples to use.

        Returns:
            The batch mean loss.
        """
        for batch_i, batch in enumerate(val_batched_pairs):
            batch_size = len(batch)
            
            targets_1_pos = [pair[0] for pair in batch]
            contexts_1_pos = [pair[1] for pair in batch]
            contexts_0_pos_samples = self.model.get_neg_samples(batch_size, num_neg_samples)

            targets_1_pos = Variable(torch.LongTensor(targets_1_pos))
            contexts_1_pos = Variable(torch.LongTensor(contexts_1_pos))
            contexts_0_pos_samples = Variable(torch.LongTensor(contexts_0_pos_samples))
            if (self.model.cuda_is_avail):
                targets_1_pos = targets_1_pos.cuda()
                contexts_1_pos = contexts_1_pos.cuda()
                contexts_0_pos_samples = contexts_0_pos_samples.cuda()         

            batch_mean_loss = self.model.forward(targets_1_pos, contexts_1_pos, contexts_0_pos_samples)
        return batch_mean_loss.data[0]