import torch.nn as nn
from config import INPUT_DIM, OUTPUT_DIM, MLP_HIDDEN

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = []
        prev_dim = INPUT_DIM
        for hidden_dim in MLP_HIDDEN:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)