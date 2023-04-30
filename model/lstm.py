# create a LSTM model
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  output_size, dropout = 0.1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size,  output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        fc_out = self.fc(out[:, -1, :])
        return fc_out, out