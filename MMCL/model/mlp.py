import torch.nn as nn
from torch.nn import LayerNorm,Linear,Dropout,Softmax

class Mlp(nn.Module):
    def __init__(self, input_dim, intermediate_dim, num_labels):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, input_dim)
        self.act_fn = nn.ReLU()
        self.dropout = Dropout(0.5)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x