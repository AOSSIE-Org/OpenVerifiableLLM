import torch

class TinyDataset:
    def __init__(self):
        self.vocab = ['a', 'b', 'c', 'd']
        self.vocab_size = len(self.vocab)

        self.data = "abcdabcdabcdabcd"

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.iots = {i: ch for ch, i in self.stoi.items()}

        self.encoded = torch.tensor(
            [self.stoi[ch] for ch in self.data],
            dtype=torch.long
        )
    
    def get_batch(self, block_size = 4):
        #for linear model: x = self.encoded[:block_size]
        x = self.encoded[:block_size].unsqueeze(0)
        # for linearmodel: y = self.encoded[1:block_size+1]
        y = self.encoded[1:block_size+1].unsqueeze(0)
        return x, y