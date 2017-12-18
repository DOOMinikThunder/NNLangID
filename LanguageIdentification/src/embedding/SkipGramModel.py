# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import unicodecsv as csv
import os
import numpy as np


class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.embed_dim = embed_dim
        
        
#    # returns the index of <num_samples> randomly picked negative samples (0-entries)
#    def negative_sampling(self, num_samples):
#        
#        return 