# create a LSTM model
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.autograd import Variable
from torch import Tensor


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
    
if __name__ == '__main__':
    model = nn.LSTM(50, 100, 2, batch_first=True)
    # x = Variable(Tensor(50, 32, 50)) # seq_len, batch, input_size
    x = Variable(Tensor(32, 50, 50)) # batch, seq_len, input_size
    #h = model.init_hidden(32)
    h = (Variable(Tensor(2*2, 32, 100)),
         Variable(Tensor(2*2, 32, 100)))
    print(model(x, h))
    print("")