# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F



class SkipGramModel(nn.Module):

    
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
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
        emb_h = self.embed_hidden(autograd.Variable(torch.LongTensor(targets)))
        emb_o = self.embed_output(autograd.Variable(torch.LongTensor(contexts)))
        score = torch.mul(emb_h, emb_o).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_o = self.embed_output(autograd.Variable(torch.LongTensor(neg_samples)))
        neg_score = torch.bmm(neg_emb_o, emb_h.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
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