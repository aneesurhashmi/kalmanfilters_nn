from torch import nn
import torch
from model import RNN , LSTM, LSTMLayerNorm
import model.gru as gru

_models = {
    "RNN": RNN,
    "LSTM": LSTM,
    "LSTM_ln": LSTMLayerNorm,
    "GRU": gru.GRU
}

class Base(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model='RNN', dropout=0.2, *args):
        super(Base, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model
        self._base = _models[model](input_size = input_size, 
                                    hidden_size = hidden_size, 
                                    num_layers = num_layers, 
                                    output_size = output_size, 
                                    dropout = dropout, *args)

    def forward(self, x):
        return self._base(x)