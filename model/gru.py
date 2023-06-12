import torch
from torch import nn
from torch.nn import functional as F
import haste_pytorch as haste
from torch.autograd import Variable
from torch import Tensor
class HasteGRU(nn.Module):
    def __init__(self, input_size,
                  hidden_size,
                  output_size, 
                  num_layers, 
                  dropout=0.2,
                  biderctional=0,
                  zoneout=0.0,
                  batch_first=True,
                    ):
        super(HasteGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.direction = biderctional+1
        

        self.gru_layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.direction
            self.gru_layers.append(haste.LayerNormGRU(layer_input_size, hidden_size, dropout=dropout, batch_first=True))
       
        self.gru_layers = nn.ModuleList(self.gru_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # h = hidden if hidden is not None else None
        h = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) if h is None else h
        
        for layer in self.gru_layers:
            x, h = layer(x, h)

        x = self.fc(x[:, -1, :])
        return x, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, hidden = self.gru(x, h0)
        output = self.fc(output[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    

if __name__ == '__main__':
    input_size = 50 # feature size 
    hidden_size = 100
    batch_size = 32
    seq_len = 50
    output_size = 3
    num_layers = 4

    model = HasteGRU(input_size=input_size, hidden_size = hidden_size, 
                                output_size = output_size, num_layers = num_layers,
                                batch_first=True)
    x = Variable(Tensor(batch_size, seq_len, input_size)) # batch, seq_len, input_size
    h = (Variable(Tensor(2*2, 32, 100)))
    # print(model(x, h))
    print(model(x))
    print("")