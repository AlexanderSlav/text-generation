import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, txt_path, mode='train', seq_len=50, n_steps=50):
        assert os.path.exists(txt_path), f"File not found {txt_path}\n"

        with open(txt_path, 'r') as txt:
            data = txt.read()

        self.int2char, self.char2int = self.get_lookup_tables(data)
        self.encoded = np.array([self.char2int[ch] for ch in data])
        self.chars = tuple(self.char2int.keys())
        self.batches = list(self.get_batches(self.encoded, seq_len, n_steps))
        self.n_symbols = len(self.chars)

    def get_lookup_tables(self, text):
        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}

        return int2char, char2int

    def __len__(self):
        return len(self.batches)

    def get_batches(self, arr, n_seqs, n_steps):
        '''Create a generator that returns batches of size
           n_seqs x n_steps from arr.
        '''

        batch_size = n_seqs * n_steps
        n_batches = len(arr) // batch_size

        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size]
        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            # The features
            x = arr[:, n:n + n_steps]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield torch.from_numpy(x), torch.from_numpy(y)

    def __getitem__(self, item):
        inp, target = self.batches[item]
        inp = F.one_hot(inp, self.n_symbols).float()
        # inp, target = torch.from_numpy(inp), torch.from_numpy(target)
        return inp, target

# TextDataset('data/arxiv_small.txt')

