
# %% [markdown]
# ## Let's try RNN

# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import random
# import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

random.seed(0)
torch.manual_seed(0)


# %%
# new data
DATA_DIR = "/home/anees.hashmi/Desktop/ML703/2D/evaluation dataset"
CSV_NAME = os.listdir(DATA_DIR)[0]

# %%
# change names
# for csv_path in os.listdir(DATA_DIR):
#     my_df = pd.read_csv(os.path.join(DATA_DIR, csv_path))
#     alpha = my_df["alpha"].unique()
#     print(alpha)
    # if csv_path.endswith(".csv"):
        # os.rename(os.path.join(DATA_DIR, csv_path), os.path.join(DATA_DIR, CSV_NAME))

# %% [markdown]
# ## Merge CSVs

# %%
df = pd.read_csv(os.path.join(DATA_DIR, f'{CSV_NAME}'))

# df = df.dropna(axis=1) # to prevent nan in model output
df.head()


# %%
x = ["ground_truth_x", "ground_truth_y", "ground_truth_theta", 
     "kalman_prediction_x", "kalman_prediction_y", "kalman_prediction_theta", 
     'ekf_pos_x', 'ekf_pos_y', 'ekf_pos_theta', 
     'ukf_pos_x', 'ukf_pos_y', 'ukf_pos_theta']

# %%
features_df = df.drop(columns=x)
features_df.head()
# features_df.columns

# %%

# %% [markdown]
# ## Columns for 2D case
# - Kalman Predictions:
#     - kalman_prediction_x
#     - kalman_prediction_y
#     - kalman_prediction_theta
# 
# - EKF:
#     - ekf_pos_x 
#     - ekf_pos_y 
#     - ekf_pos_theta
# 
# - UKF:
#     - ukf_pos_x 
#     - ukf_pos_y 
#     - ukf_pos_theta
# 
# 
# - Ground Truth:
#     - ground_truth_x
#     - ground_truth_y
#     - ground_truth_theta

# %%
# Split
# train_df = df.sample(frac=0.8, random_state=0)
# test_df = df.drop(df.index)


# sanity checks
# kf_pred = torch.tensor(df['kalman prediction'].to_list()).reshape(-1, 1)
# gt = torch.tensor(df['ground truth'].to_list()).reshape(-1, 1)

# split = -1


# plt.figure(figsize=(20, 10))
# plt.plot(df['ground_truth_x'].to_list()[:split])
# plt.plot(df['ground_truth_y'].to_list()[:split])
# plt.plot(df['ground_truth_theta'].to_list()[:split])

# plt.plot(df['kalman_prediction_x'].to_list()[:split])
# plt.plot(df['kalman_prediction_y'].to_list()[:split])
# plt.plot(df['kalman_prediction_theta'].to_list()[:split])

# plt.plot(df['ekf_pos_x'].to_list()[:split])
# plt.plot(df['ekf_pos_y'].to_list()[:split])
# plt.plot(df['ekf_pos_theta'].to_list()[:split])

# plt.legend(['ground truth x', 'ground truth y', 'ground truth theta','kalman_prediction_x', 'kalman_prediction_y', 'kalman_prediction_theta', 'ekf_pos_x', 'ekf_pos_y', 'ekf_pos_theta'])
# # plt.legend(['kalman prediction x', 'ground truth x', "kalman_pred_y","ground_truth_y", "kalman_prediction_theta",  "ground_truth_theta"])
# plt.show()

# %%
# df[df.isna().values]

# %% [markdown]
# ## Model

# %%
# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1 ):
        super(SimpleRNN, self).__init__()
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

# %% [markdown]
# ## Reshape csv as per the seq length and batch size

# %%
def get_input_data(df, seq_len, batch_size, num_batches):

       labels = df[["ground_truth_x", "ground_truth_y", "ground_truth_theta" ]].to_numpy()
       kalman_pred = df[["kalman_prediction_x", "kalman_prediction_y", "kalman_prediction_theta"]].to_numpy() # used for comparison
       ekf_pred = df[["ekf_pos_x", "ekf_pos_y", "ekf_pos_theta"]].to_numpy() # used for comparison
       ukf_pred = df[["ukf_pos_x", "ukf_pos_y", "ukf_pos_theta"]].to_numpy() # used for comparison

       
       input_data_df = df[
              ['noisy_motion_x', 'noisy_motion_y', 'noisy_motion_theta',
              'noisy_motion_cov_xx', 'noisy_motion_cov_xy', 'noisy_motion_cov_xtheta',
              'noisy_motion_cov_yy', 'noisy_motion_cov_ytheta',
              'noisy_motion_cov_thetatheta', 'lidar_x', 'lidar_y', 'lidar_theta',
              'lidar_cov_xx', 'lidar_cov_xy', 'lidar_cov_xtheta', 'lidar_cov_yy',
              'lidar_cov_ytheta', 'lidar_cov_thetatheta', 'alpha']
       ].to_numpy()

       # input_data = input_data_df[:num_batches * batch_size]
       input_data = input_data_df.copy()

       # output_data = labels[:num_batches * batch_size]
       output_data = labels.copy()



       new_input_data = []
       new_output_data = []
       new_kp_data = []
       new_ekf_data = []
       new_ukf_data = []

       for i in range(len(input_data) - seq_len):
              new_input_data.append(input_data[i:i+seq_len])
              new_output_data.append(output_data[i + seq_len -1])
              new_kp_data.append(kalman_pred[i + seq_len -1])
              new_ekf_data.append(ekf_pred[i + seq_len -1])
              new_ukf_data.append(ukf_pred[i + seq_len -1])

       new_input_data = np.array(new_input_data[:num_batches* batch_size]) # drop the last batch
       new_output_data = np.array(new_output_data[:num_batches* batch_size]) # drop the last batch
       new_kp_data = np.array(new_kp_data[:num_batches* batch_size]) # drop the last batch
       new_ekf_data = np.array(new_ekf_data[:num_batches* batch_size]) # drop the last batch
       new_ukf_data = np.array(new_ukf_data[:num_batches* batch_size]) # drop the last batch

       return new_input_data, new_output_data, new_kp_data, new_ekf_data, new_ukf_data


# %%
seq_len = 10
batch_size = 2
num_batches = (len(df) - seq_len) // (batch_size)

new_input_data, new_output_data, kalman_pred_data, ekf_pred_data, ukf_pred_data =  get_input_data(df, seq_len = seq_len, batch_size = batch_size, num_batches = num_batches)


# new_input_data.shape == (num_batches* batch_size, seq_len, new_input_data.shape[-1])

# %%
# new_input_data.shape, new_output_data.shape, kalman_pred_data.shape,  ekf_pred_data.shape, ukf_pred_data.shape

# %%
# # sanity check
# print(new_input_data[0][-1], new_output_data[0].round(6))
print(df.iloc[seq_len - 1][["ground_truth_x", "ground_truth_y", "ground_truth_theta" ]].to_numpy(), new_output_data[0])
# == new_output_data[0].round(6)

# df.iloc[seq_len - 1]

# %%
# split
# train_input_data = new_input_data[:num_batches* batch_size//2]
# train_output_data = new_output_data[:num_batches* batch_size//2]
# new_input_data[:len(new_input_data)*0.8, ].shape

test_idx = random.sample(range(len(new_input_data)), k = int(len(new_input_data)*0.2))
train_idx = [i for i in range(len(new_input_data)) if i not in test_idx]

test_input = new_input_data[test_idx]
test_output = new_output_data[test_idx]

train_input = new_input_data[train_idx]
train_output = new_output_data[train_idx].reshape(-1, 3)

train_kalman_pred = kalman_pred_data[train_idx]
test_kalman_pred = kalman_pred_data[test_idx]

train_ekf_pred = ekf_pred_data[train_idx]
test_ekf_pred = ekf_pred_data[test_idx]

train_ukf_pred = ukf_pred_data[train_idx]
test_ukf_pred = ukf_pred_data[test_idx]

# %%
# train_input.shape, train_output.shape, test_input.shape, test_output.shape, train_kalman_pred.shape, test_kalman_pred.shape, train_ekf_pred.shape, test_ekf_pred.shape, train_ukf_pred.shape, test_ukf_pred.shape

# %%
# train_input.shape, train_output.shape, test_input.shape, test_output.shape

# train_input.shape, batch_size, (train_input.shape[0]//batch_size)*batch_size
# 

# get pytorch csv dataloader
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False , drop_last=True)

# %% [markdown]
# ## Gear Up | Train | Launch

# %%
input_size = train_input.shape[2]
output_size = train_output.shape[1]
hidden_size = 256
num_layers = 2
lr = 0.0001

# writer = SummaryWriter(f"runs/RNN/seq_len_{seq_len}_batch_size_{batch_size}_hidden_size_{hidden_size}_lr_{lr}")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = SimpleRNN(input_size=input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size).to(device)

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# criterion_l1 = nn.L1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.99)
# # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.0001, 0.0001, 1000)


# # %%
# # Train the model
# loss_cache = []


# num_epochs = 1
# for epoch in range(num_epochs):
#     for inputs, targets in train_loader:
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         optimizer.zero_grad()
#         output, hidden_output = model(inputs)
#         loss = criterion(output, targets)
#         loss.backward()
#         optimizer.step()
#         # lr_scheduler.step()

#     writer.add_scalar('training loss', loss, epoch)
#     loss_cache.append(loss.item())
#     # Print the loss after every epoch
#     if epoch %5 ==0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



# # %%
# plt.plot(loss_cache)

# # %% [markdown]
# # ## Pred

# # %%
# loss_cache = []
# test_pred = []
# with torch.no_grad():
#     for inputs, outputs in test_loader:
#         inputs = inputs.to(device)
#         outputs = outputs.to(device).view(-1, 3)
#         # outputs = outputs
#         pred, _ = model(inputs)
#         test_pred.append(pred)
#         loss = criterion(pred, outputs)
#         # print(loss.item())
#         loss_cache.append(loss.item())

# # %%
# # Kalman filter prediction loss
# loss_kp = criterion(torch.from_numpy(test_kalman_pred), torch.from_numpy(test_output))

# loss_ekf = criterion(torch.from_numpy(test_ekf_pred), torch.from_numpy(test_output))
# print("Test MSE RNN: {:.4f}".format(sum(loss_cache)))
# print("Test MSE Kalman: {:.4f}".format(loss_kp))
# print("Test MSE EKF: {:.4f}".format(loss_ekf))


# # %%
# # torch.stack(test_pred).reshape(-1, 3).shape

# # %%
# # test_output.shape, test_pred.shape
# test_pred_y = torch.stack(test_pred).reshape(-1, 3)[:, 1].cpu().numpy()
# test_pred_x = torch.stack(test_pred).reshape(-1, 3)[:, 0].cpu().numpy()
# test_pred_theta = torch.stack(test_pred).reshape(-1, 3)[:, 2].cpu().numpy()

# test_output_x = test_output[:, 0]
# test_output_y = test_output[:, 1]
# test_output_theta = test_output[:, 2]

# kalman_pred_x = test_kalman_pred[:, 0]
# kalman_pred_y = test_kalman_pred[:, 1]
# kalman_pred_theta = test_kalman_pred[:, 2]

# ekf_pred_x = test_ekf_pred[:, 0]
# ekf_pred_y = test_ekf_pred[:, 1]
# ekf_pred_theta = test_ekf_pred[:, 2]

# ukf_pred_x = test_ukf_pred[:, 0]
# ukf_pred_y = test_ukf_pred[:, 1]
# ukf_pred_theta = test_ukf_pred[:, 2]

# # %%
# split = 100


# plt.figure(figsize=(20, 10))

# plt.subplot(3, 1, 1)
# plt.plot(test_pred_x[:split])
# plt.plot(test_output_x[:split])
# plt.plot(kalman_pred_x[:split])
# plt.legend(['rnn prediction', 'ground truth', 'kalman prediction'])
# plt.title("X")# %% [markdown].


# plt.subplot(3, 1, 2)
# plt.plot(test_pred_y[:split])
# plt.plot(test_output_y[:split])
# plt.plot(kalman_pred_y[:split])
# plt.legend(['rnn prediction', 'ground truth', 'kalman prediction'])
# plt.title("Y")

# plt.subplot(3, 1, 3)
# plt.plot(test_pred_theta[:split])
# plt.plot(test_output_theta[:split])
# plt.plot(kalman_pred_theta[:split])
# plt.legend(['rnn prediction', 'ground truth', 'kalman prediction'])
# plt.title("Theta")



# # %%
# split = 200


# plt.figure(figsize=(20, 10))

# plt.subplot(3, 1, 1)
# plt.plot(test_pred_x[:split])
# plt.plot(test_output_x[:split])
# plt.plot(ekf_pred_x[:split])
# plt.legend(['rnn prediction', 'ground truth', 'EKF prediction'])
# plt.title("X")


# plt.subplot(3, 1, 2)
# plt.plot(test_pred_y[:split])
# plt.plot(test_output_y[:split])
# plt.plot(ekf_pred_y[:split])
# plt.legend(['rnn prediction', 'ground truth', 'EKF prediction'])
# plt.title("Y")

# plt.subplot(3, 1, 3)
# plt.plot(test_pred_theta[:split])
# plt.plot(test_output_theta[:split])
# plt.plot(ekf_pred_theta[:split])
# plt.legend(['rnn prediction', 'ground truth', 'EKF prediction'])
# plt.title("Theta")



# %%
# def save_checkpoint(checkpoint):
#     model_save_dir = f'./results/{CSV_NAME}'
#     # print(model_save_dir)
#     os.makedirs(model_save_dir, exist_ok=True)
#     # chckpoint = {
#     #     'epoch': num_epochs,
#     #     'model_state_dict': model.state_dict(),
#     #     'optimizer_state_dict': optimizer.state_dict(),
#     #     'loss': torch.tensor(loss_cache),
#     # }

#     torch.save(checkpoint, os.path.join(model_save_dir, "model.pt"))

# %% [markdown]
# ## Setting up Ray Tune

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import ray
from ray import tune

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# %%
num_epochs = 5

# Define the training function
def train_rnn(config):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            # net = nn.DataParallel(net)
            pass
    model = SimpleRNN(input_size=config["input_size"], hidden_size = config["hidden_size"], 
                      num_layers = config["num_layers"], output_size = config["output_size"])
    # model = RNNModel(config["input_size"], config["hidden_size"], config["num_layers"], config["output_size"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            inputs = inputs
            targets = targets

            optimizer.zero_grad()
            output, hidden_output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()


    test_pred = []
    loss_cache = []
    with torch.no_grad():
        for inputs, outputs in test_loader:
            inputs = inputs
            outputs = outputs.view(-1, 3)
            # outputs = outputs
            pred, _ = model(inputs)
            test_pred.append(pred)
            loss = criterion(pred, outputs)
            loss_cache.append(loss.item())
            # print(loss.item())

    # accuracy = 100 * correct / total
    # inverse_MSE = len(loss_cache) / sum(loss_cache) # using inverse MSE as the metric
    tune.report(loss=(sum(loss_cache) / len(loss_cache)))



def main(num_samples=1, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.path.abspath("./data")
    # load_data(data_dir)
    config = {
        "input_size": tune.choice([input_size]),
        "hidden_size": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([1]),
        "output_size": [train_output.shape[1]],
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": 10
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        metric_columns=["loss"])
    result = tune.run(
        train_rnn,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = SimpleRNN(input_size=config["input_size"], hidden_size = config["hidden_size"], 
                      num_layers = config["num_layers"], output_size = config["output_size"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

# # Set up data loaders
# # train_loader, val_loader, test_loader = get_data_loaders()

# # Set up device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# # %%

# # # Set up Ray Tune
# # try:
# # ray.init()
# # if __name__ == '__main__':
# ray.init(num_gpus=1, num_cpus=16)
# # run_search()
# # except:
# #     pass
# analysis = tune.run(train_rnn, config=config, num_samples=10)
# num_samples = 10
# scheduler = ASHAScheduler(

# gpus_per_trial = 2
# result = tune.run(
#     train_rnn,
#     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#     config=config,
#     num_samples=num_samples,
#     scheduler=scheduler,
#     progress_reporter=reporter,
#     checkpoint_at_end=True)