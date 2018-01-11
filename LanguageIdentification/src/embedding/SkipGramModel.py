# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F



"""
Loss calculation in forward method is based on the Negative sampling objective (NEG) (formula 4)
in paper "Distributed Representations of Words and Phrases and their Compositionality"
by Tomas Mikolov et al. (Oct. 2013).
Initializations and optimizations are suggested by Xiaofei Sun on
https://adoni.github.io/2017/11/08/word2vec-pytorch/ (Access: 11.01.2018)
"""
class SkipGramModel(nn.Module):

    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_hidden = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.embed_output = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.init_embed()
        
        
    def init_embed(self):
        init_range = 0.5 / self.embed_dim
        self.embed_hidden.weight.data.uniform_(-init_range, init_range)
        self.embed_output.weight.data.uniform_(-0, 0)


    def forward(self, targets_1_pos, contexts_1_pos, contexts_0_pos_samples):
        losses = []
        # lookup the 1-position weight values for the target char
        # for all target chars in the batch
        targets_1_pos_weights_hidden = self.embed_hidden(autograd.Variable(torch.LongTensor(targets_1_pos)))
#        print(targets_1_pos_weights_hidden)
        # lookup the 1-position weight values for the context char (backwards from "output layer")
        # for all context chars in the batch
        contexts_1_pos_weights_output = self.embed_output(autograd.Variable(torch.LongTensor(contexts_1_pos)))
#        print(contexts_1_pos_weights_output)
        # calculate dot product for each target_1_pos-context_1_pos pair in the batch
        score_contexts_1_pos = torch.mul(targets_1_pos_weights_hidden, contexts_1_pos_weights_output)
#        print(score_contexts_1_pos)
        score_contexts_1_pos = torch.sum(score_contexts_1_pos, dim=1)
#        print(score_contexts_1_pos)
        # apply log sigmoid function to the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_1_pos = F.logsigmoid(score_contexts_1_pos)
#        print(score_contexts_1_pos)
        losses.append(sum(score_contexts_1_pos))
        # use the sampled 0-positions of the context char to lookup the weight values (backwards from "output layer")
        # for all context chars in the batch
        contexts_0_pos_samples_weights_output = self.embed_output(autograd.Variable(torch.LongTensor(contexts_0_pos_samples)))
#        print(contexts_0_pos_samples_weights_output)
        # calculate dot product for each target_1_pos-context_0_pos_sample pair
        # for the whole batch
        score_contexts_0_pos_samples = torch.bmm(contexts_0_pos_samples_weights_output, targets_1_pos_weights_hidden.unsqueeze(2))
#        print(score_contexts_0_pos_samples)
        score_contexts_0_pos_samples = torch.sum(score_contexts_0_pos_samples, dim=1)
#        print(score_contexts_0_pos_samples)
        # apply log sigmoid function to the negative of the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score_contexts_0_pos_samples = F.logsigmoid(-1 * score_contexts_0_pos_samples)
#        print(score_contexts_0_pos_samples)
        losses.append(sum(score_contexts_0_pos_samples))
        # sum up the score_contexts_1_pos and score_contexts_0_pos_samples,
        # make positive (is negative because of log sigmoid function)
        # and normalize the loss by dividing by the batch size
#        print(losses)
        return (-1 * sum(losses)) / len(targets_1_pos)
    
    
    def save_embed_to_file(self, relative_path_to_file):
        weights_array = self.embed_hidden.weight.data.numpy()
        writer = open(relative_path_to_file, 'w')
        # write vocabulary size and embedding dimension to file
        writer.write('%d %d' % (self.vocab_size, self.embed_dim))
        # write weights to file (one row for each char of the vocabulary)
        for i in range(self.vocab_size):
            line = ' '.join(map(lambda x: str(x), weights_array[i]))
            writer.write('\n%s' % line)
        print('Embedding weights saved to file:', relative_path_to_file)