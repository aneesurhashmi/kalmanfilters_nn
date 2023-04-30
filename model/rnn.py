import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1,*args):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    # def forward(self, x, h0):
    def forward(self, x):
        # h0 from the last batch

        # print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print(h0.shape, torch.zeros(1, x.size(0), self.hidden_size).to(x.device).shape)
        out, _ = self.rnn(x, h0)
        if out.isnan().any():
            print("NAN:", out)
        fc_out = self.fc(out[:, -1, :]) # taking only the last hidden layer output
        # print(fc_out.shape, out.shape)
        if out.isnan().any():
            print("NAN: Second: ", out)
        return fc_out, out
    

# class RNNModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNNModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         out, _ = self.rnn(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out