import torch.nn as nn
import torch


class BinaryClassifier(nn.Module):
    def __init__(self, n_latent):
        super(BinaryClassifier, self).__init__()
        self.lr = 0.001
        self.weight_decay = 0
        self.batch_size = 10
        self.layer1 = nn.Linear(n_latent, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x
    
    