import numpy as np

class LSA:
    def __init__(self, vocab_map, documents):
        self.W = np.zeros(len)