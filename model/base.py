from torch import nn
import torch
from model import RNN , LSTM, LSTMLayerNorm, HasteLSTMLayerNorm
import model.gru as gru
import os

_models = {
    "RNN": RNN,
    "LSTM": LSTM,
    "LSTM_ln": HasteLSTMLayerNorm,
    "GRU": gru.HasteGRU
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
    
    def load_chkpt(self, config):

        if os.path.isdir(config['dir']):
            chkpoints_list = [i for i in os.listdir(config['dir']) if i.startswith("checkpoint")]
            last_checkpoint = sorted(chkpoints_list, key=lambda x: int(x.split('_')[-1]))[-1]
            chkpoint_state_dict, optim = torch.load(os.path.join(config['dir'], last_checkpoint, 'checkpoint'))

            path = os.path.join(config['dir'], last_checkpoint, 'checkpoint')
        else:
            # chkpoint_state_dict = torch.load(os.path.join(config['dir']))
            chkpoint_state_dict = torch.load(os.path.join(config['dir']))
            path = os.path.join(config['dir'])

        self.load_state_dict(chkpoint_state_dict)
        return path
    
def make_model(cfg, config=None):

    model =  Base(input_size = cfg.MODEL.INPUT_SIZE, 
                hidden_size = cfg.MODEL.HIDDEN_SIZE, 
                num_layers = cfg.MODEL.NUM_LAYERS, 
                output_size = cfg.MODEL.OUTPUT_SIZE, 
                model = cfg.MODEL.TYPE, 
                dropout = cfg.MODEL.DROPOUT)
    if config is not None:
        model = Base(input_size=cfg.MODEL.INPUT_SIZE,
                hidden_size = config["hidden_size"],
                num_layers = config["num_layers"],
                output_size=1 if cfg.DATA.SETTING == '1D' else 3,
                model = cfg.MODEL.TYPE,
                dropout=cfg.MODEL.DROPOUT)
        path = model.load_chkpt(config)
    else:
        path = None

    return model, path
