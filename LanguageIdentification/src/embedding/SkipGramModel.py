# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F



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


    def forward(self, targets, contexts, neg_samples):
        losses = []
        # lookup the weight values for the target char
        # for all target chars in the batch
        emb_h = self.embed_hidden(autograd.Variable(torch.LongTensor(targets)))
        # lookup the 1-position weight value for the context char (backwards from "output layer")
        # for all context chars in the batch
        emb_o = self.embed_output(autograd.Variable(torch.LongTensor(contexts)))
        # calculate dot product for each target-1_pos_context pair in the batch
        score = torch.mul(emb_h, emb_o).squeeze()
        score = torch.sum(score, dim=1)
        # apply log sigmoid function to the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        score = F.logsigmoid(score)
        losses.append(sum(score))
        # use the sampled 0-positions of the context char to lookup the weight values (backwards from "output layer")
        neg_emb_o = self.embed_output(autograd.Variable(torch.LongTensor(neg_samples)))
        # calculate dot product for each target-0_pos_context pair
        # for the whole batch
        neg_score = torch.bmm(neg_emb_o, emb_h.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        # apply log sigmoid function to the negative of the calculated dot products in the batch
        # and sum up the results for the whole batch and store in list
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        # sum up the score and neg_score, negate and normalize the loss by dividing by the batch size
        return (-1 * sum(losses)) / len(targets)
    
    
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