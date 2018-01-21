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


import math

class Batches(object):
    def __init__(self, data_set, targets, batch_size):
        """
        Iterator class to ease the use of mini-batches
        Args:
        	data_set: will be divided into batches
        	targets: will be divided into batches
        	batch_size: size each batch of input and target will have
        """
        self.data_set = data_set
        self.targets = targets
        self.batch_size = batch_size
        self.num_batches = int(math.floor(len(self.data_set)/self.batch_size)) #will ignore the last few dates too small to form a batch

    def __iter__(self):
        """

        Returns: iterator over all batches

        """
        for batch in range(self.num_batches):
            yield self.data_set[batch*self.batch_size:(batch+1)*self.batch_size], self.targets[batch*self.batch_size:(batch+1)*self.batch_size]
