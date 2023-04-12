# %% [markdown]
# ## Let's try RNN

# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import shutil
from torch.utils.data import random_split

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

random.seed(0)
torch.manual_seed(0)


# %%
DATA_DIR = "./generated_data"
CSV_NAME = "mnst_2_lnstd_0.1.csv"



# %%
# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    # def forward(self, x, h0):
    def forward(self, x):
        # h0 from the last batch

        # print(x.shape)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # print(h0.shape, torch.zeros(1, x.size(0), self.hidden_size).to(x.device).shape)
        out, _ = self.rnn(x, h0)
        if out.isnan().any():
            print("NAN:", out)
        fc_out = self.fc(out[:, -1, :]) # taking only the last hidden layer output
        # print(fc_out.shape, out.shape)
        if out.isnan().any():
            print("NAN: Second: ", out)
        return fc_out, out


def get_input_data(train_df, seq_len, batch_size, num_batches):

       labels = np.array(train_df["ground truth"].to_list())
       kalman = np.array(train_df['kalman prediction'].to_list())
       input_data_df = train_df[['noisy motion', 'motion noise stdev', 'laser noise stdev', 'laser range 1',
              'laser range 2', 'laser range 3', 'laser range 4', 'laser range 5',
              'laser range 6']].to_numpy()

       # input_data = input_data_df[:num_batches * batch_size]
       input_data = input_data_df.copy()

       # output_data = labels[:num_batches * batch_size]
       output_data = labels.copy()



       new_input_data = []
       new_output_data = []
       new_kalman = []
       for i in range(len(input_data) - seq_len):
              new_input_data.append(input_data[i:i+seq_len])
              new_output_data.append(output_data[i + seq_len -1])
              new_kalman.append(kalman[i + seq_len - 1])

       new_input_data = np.array(new_input_data[:num_batches* batch_size]) # drop the last batch
       new_output_data = np.array(new_output_data[:num_batches* batch_size]) # drop the last batch
       new_kalman = np.array(new_kalman[:num_batches* batch_size]) # drop the last batch
       
       return new_input_data, new_output_data, new_kalman



def load_data(datapath, test_split=0.15, seq_len = 100, batch_size = 10):

    df = pd.read_csv(datapath)
    df = df.dropna(axis=1) # to prevent nan in model output

    seq_len = 100
    batch_size = 10
    num_batches = (len(df) - seq_len) // (batch_size)
    new_input_data, new_output_data, kalman_pred = get_input_data(df, seq_len = seq_len, batch_size = batch_size, num_batches = num_batches)
    
    test_idx = random.sample(range(len(new_input_data)), k = int(len(new_input_data)*test_split))
    train_idx = [i for i in range(len(new_input_data)) if i not in test_idx]

    test_input = new_input_data[test_idx]
    test_output = new_output_data[test_idx]
    test_kalman = kalman_pred[test_idx]

    train_input = new_input_data[train_idx]
    train_output = new_output_data[train_idx].reshape(-1, 1)
    train_kalman = kalman_pred[train_idx].reshape(-1, 1)

    # get pytorch csv dataloader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_input).float(), torch.from_numpy(test_output).float())
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False , drop_last=True)

    return train_dataset, test_dataset, test_kalman, train_kalman, 


# train_dataset, test_dataset, test_kalman, train_kalman = load_data(datapath=datapath)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = SimpleRNN(input_size, hidden_size, output_size).to(device)

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# criterion_l1 = nn.L1Loss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# # Train the model
# loss_cache = []

def train(config, checkpoint_dir=None, data_dir=None):

    input_size = 9
    output_size = 1
    hidden_size = 256
    lr = 0.0001
    net = SimpleRNN(input_size, config["hidden_size"], output_size)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.MSELoss()
    # criterion_l1 = nn.L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset, _, _ = load_data(data_dir, test_split=0.15, seq_len=config['seq_len'], batch_size=config['batch_size'])

    test_abs = int(len(trainset) * 0.85)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden_output = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, hidden_output = net(inputs)
                total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")

def test_accuracy(net,data_dir, device="cpu"):
    trainset, testset,_,_ = load_data(data_dir)
    criterion = nn.MSELoss()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    loss = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)

            predicted, hidden_out = net(input)

            loss = criterion(predicted, labels)

            total += labels.size(0)

    return loss / total

def plot_result(test_kalman, test_output, test_pred):
    split = 50
    plt.figure(figsize=(20, 10))
    plt.plot(test_kalman[:split])
    plt.plot(test_output[:split])
    # plt.plot(test_pred.detach().cpu()[:split])

    plt.legend(['kalman prediction', 'ground truth', 'rnn prediction'])
    # plt.legend(['ground truth', 'rnn prediction'])
    # plt.plot(pred.to_list())

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # data_dir = os.path.abspath("./data")
    data_dir = os.path.abspath(os.path.join(DATA_DIR, CSV_NAME))
    # load_data(data_dir)

    input_size = 9
    output_size = 1
    hidden_size = 256
    lr = 0.0001
    
    config = {
    "hidden_size": tune.sample_from(lambda _: 2**np.random.randint(2, 8)),
    "seq_len": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

    best_trained_model = SimpleRNN(input_size, config["hidden_size"], output_size)
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

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
    # data_dir = os.path.join(DATA_DIR, CSV_NAME)
    # te, tr, kal, tes = load_data(datapath = data_dir)
# if __name__ =='__main__':

#     datapath = os.path.join(DATA_DIR, CSV_NAME)
#     input_size = 9
#     output_size = 1
#     hidden_size = 256
#     lr = 0.0001

#     config = {
#     "hidden_size": tune.sample_from(lambda _: 2**np.random.randint(2, 8)),
#     "seq_len": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([2, 4, 8, 16])
#     }

#     gpus_per_trial = 2
#     # ...
#     result = tune.run(
#         partial(train, data_dir=data_dir),
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         checkpoint_at_end=True) 
