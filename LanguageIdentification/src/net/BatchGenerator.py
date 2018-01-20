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
