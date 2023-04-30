import json
import ray
from ray import tune
from model import RNN, RNNModel, Base
from utils import plot_data, get_input_data, get_dataloader
import os
import torch
from torch import nn
import numpy as np
TRAIN_DATA_DIR = './data/2D/generated_data'
EVAL_DATA_DIR = './data/2D/evaluation_data'

# model dictionary
_models = ["SimpleRNN", "RNNModel"]

params = {
        "batch_size": 100,
        "model_name": "SimpleRNN",
        "input_size": 19,
        "output_size": 3,
        "num_epochs": 100,
        "log_step": 500,
        'TRAIN_DATA_DIR': TRAIN_DATA_DIR,
    }

# best configuration
with open('logs/best_config_2D.json', 'r') as fp:
    best_config = json.load(fp)

#setup device
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:0"

def test_accuracy(net, test_loader, device="cpu"):
    
    net.to(device)
    criterion = nn.MSELoss()
    loss = 0
    total = 0
    test_pred = []
    with torch.no_grad():
        for data in test_loader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            outputs, _ = net(input)
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss += criterion(outputs, labels).cpu().numpy()
            test_pred.append(outputs.cpu().numpy())
    return loss / total, test_pred

# data
X_test, y_test, y_kalman_test, y_ekf_test, y_ukf_test = get_input_data(seq_len = best_config['sequence_length'], batch_size = params['batch_size'], datadir=EVAL_DATA_DIR)
test_loader = get_dataloader(X_test,y_test, batch_size=params['batch_size'])

best_trained_model = Base(input_size=params['input_size'], hidden_size = best_config["hidden_size"],
                                                       num_layers = best_config["num_layers"], output_size = params['output_size'], model=params['model_name'])

model_dir = os.path.join(best_config['dir'], os.listdir(best_config['dir'])[0] )   
model_state, optimizer_state = torch.load(os.path.join(model_dir,'checkpoint'))
best_trained_model.load_state_dict(model_state)

loss, test_pred = test_accuracy(best_trained_model, test_loader, device=device)
print("Best trial test set loss: {}".format(loss))

test_pred = np.concatenate(test_pred, axis=0)

to_plot = {
    "y_test": y_test,
    "test_pred": test_pred,
    "y_kalman_test": y_kalman_test,
    "y_ekf_test": y_ekf_test,
    "y_ukf_test": y_ukf_test
}

plot_data(to_plot)


