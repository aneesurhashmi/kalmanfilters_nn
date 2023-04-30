from torch import nn
import torch
from model import RNN , LSTM

_models = {
    "RNN": RNN,
    "LSTM": LSTM
}

class Base(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model='RNN', *args):
        super(Base, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model
        self._base = _models[model](input_size, hidden_size, num_layers, output_size, *args)

    def forward(self, x):
        return self._base(x)