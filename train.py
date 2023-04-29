import argparse
import torch
from torch import nn
from model import SimpleRNN, RNNModel, Base
from torch import optim
import os
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray
from ray import tune, air
from utils import train_test_split, get_dataloader, get_input_data, plot_data, get_input_data_1D
import numpy as np
import json
import random
import argparse
from config import cfg

random.seed(0)
torch.manual_seed(0)

def train_ray(config,cfg):

    device="cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
<<<<<<< Updated upstream
    # setup data
    print("current working directory: {}".format(os.getcwd()))
    if cfg.DATA.SETTING == '2D':
        X, y, _ =  get_input_data(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    elif cfg.DATA.SETTING == '1D':
        X, y, _ =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    else:
        raise ValueError("Environment not supported")
    
    train_data, valid_data = train_test_split(X, y, test_size=0.2)
    train_loader = get_dataloader(train_data[0],train_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)
    valid_loader = get_dataloader(valid_data[0],valid_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)
=======
    # X contains all the data (input, output, kalman, ekf, ukf)
    X = get_input_data(seq_len = config['sequence_length'], batch_size = params['batch_size'], datadir=params['TRAIN_DATA_DIR'], csv_name=params['TRAIN_DATA_CSV_NAME'])
    train_data, valid_data = train_test_split(X, test_size=0.2)
    train_loader = get_dataloader(train_data[0],train_data[1], batch_size=params['batch_size'])
    valid_loader = get_dataloader(valid_data[0],valid_data[1], batch_size=params['batch_size'])
>>>>>>> Stashed changes

    # setup model
    print("Using model: {}".format(cfg.MODEL.TYPE))
    model = Base(input_size=cfg.MODEL.INPUT_SIZE, hidden_size = config["hidden_size"], 
                      num_layers = config["num_layers"], output_size = cfg.MODEL.OUTPUT_SIZE, model=cfg.MODEL.TYPE)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(cfg.SOLVER.NUM_EPOCHS):

        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % cfg.SOLVER.LOG_STEP == 0:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _ = model(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))

    print("Finished Training")


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

<<<<<<< Updated upstream
def main(cfg):
=======
def main(cmd_args):
>>>>>>> Stashed changes

    # hyperparameter
    # num_epochs = 300
    # batch_size = 100
    num_epochs = cmd_args['num_epochs']
    batch_size = cmd_args['batch_size']

    gpus_per_trial = 1

    TRAIN_DATA_DIR = os.path.abspath("/home/anees.hashmi/Desktop/ML703/data/2D/generated_data") 
    EVAL_DATA_DIR = os.path.abspath('/home/anees.hashmi/Desktop/ML703/data/2D/evaluation_data')  

    # loop over this for multiple files
    # TRAIN_DATA_CSV_PATH = os.listdir(TRAIN_DATA_DIR)[1]
    # TRAIN_DATA_CSV_NAME = 'generated_data_fbcampus_2.csv'
    TRAIN_DATA_CSV_NAME = cmd_args['train_csv']

    # configurable parameters
    config = {
        "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_layers": tune.choice([1, 2, 3]),
        'sequence_length': tune.choice([10, 32, 64, 50]),
    }


    # # configurable parameters
    # config = {
    #     "hidden_size": 2 ** np.random.randint(2, 9),
    #     "lr": random.choice([1e-4, 1e-2]),
    #     "num_layers": random.choice([1, 2, 3]),
    #     'sequence_length': random.choice([28, 56, 112]),
    # }
    # fixed parameters
    params = {
        "batch_size": batch_size,
        "model_name": "SimpleRNN",
        "input_size": cmd_args['input_size'],
        "output_size": cmd_args['output_size'],
        "num_epochs": cmd_args['epochs'],
        "log_step": 100,
        'TRAIN_DATA_DIR': TRAIN_DATA_DIR,
        'TRAIN_DATA_CSV_NAME': TRAIN_DATA_CSV_NAME,
        'setting': '2D',
        'EVAL_DATA_DIR': EVAL_DATA_DIR,
        'environment': 'fbcampus',

    }

    # # setup device
    # device = 'cpu'
    # if torch.cuda.is_available():
    #     device = "cuda:0"


    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=10,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_ray, cfg=cfg), {'cpu':32, 'gpu':gpus_per_trial}),
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=scheduler,
        ),
        param_space=config,
        # run_config=air.RunConfig(local_dir="./logs", name="first_run")  
    )
    
    results = tuner.fit()

    best_trial = results.get_best_result("loss", "min", "last")
    best_trial.config['dir'] = str(best_trial.log_dir)
    best_trial.config["metric"] = best_trial.metrics["loss"]

    # save best config as json
<<<<<<< Updated upstream
    with open('{}/best_config_{}.json'.format(cfg.OUTPUT.OUTPUT_DIR, cfg.DATA.TRAIN_DATA_DIR.split('/')[-1][:-4]), 'w') as fp:
=======
    os.makedirs('logs', exist_ok=True)
    with open('logs/best_config_{}_{}.json'.format(params['setting'], TRAIN_DATA_CSV_NAME), 'w') as fp:
>>>>>>> Stashed changes
        json.dump(best_trial.config, fp, sort_keys=True, indent=4)
        # json.dump(best_trial.metrics, fp, sort_keys=True, indent=4)
        # json.dump({'dir':str(best_trial.log_dir)}, fp, sort_keys=True, indent=4)

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))
    
    # test on evaluation data
    # X_test, y_test, y_kalman_test, y_ekf_test, y_ukf_test = get_input_data(seq_len = best_trial.config['sequence_length'], batch_size = params['batch_size'], datadir=EVAL_DATA_DIR)
    # test_loader = get_dataloader(X_test,y_test, batch_size=params['batch_size'])

    # best_trained_model = _models[params['model_name']](input_size=params['input_size'], hidden_size = best_trial.config["hidden_size"],
    #                                                    num_layers = best_trial.config["num_layers"], output_size = params['output_size'])
    
    # best_checkpoint_dir = best_trial.log_dir
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # loss, test_pred = test_accuracy(best_trained_model, test_loader, device=device)
    # print("Best trial test set loss: {}".format(loss))

    # to_plot = {
    #     "y_test": y_test,
    #     "test_pred": test_pred,
    #     "y_kalman_test": y_kalman_test,
    #     "y_ekf_test": y_ekf_test,
    #     "y_ukf_test": y_ukf_test
    # }

    # plot_data(to_plot)

if __name__ == "__main__":
<<<<<<< Updated upstream
    parser = argparse.ArgumentParser(description="Neural networks for robot state estimation")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Get absolute path for data and output directory
    cfg.DATA.TRAIN_DATA_DIR = os.path.abspath(cfg.DATA.TRAIN_DATA_DIR)
    cfg.DATA.EVAL_DATA_DIR = os.path.abspath(cfg.DATA.EVAL_DATA_DIR)
    cfg.OUTPUT.OUTPUT_DIR = os.path.abspath(cfg.OUTPUT.OUTPUT_DIR)
    cfg.freeze()
    
    print("running with config:\n{}".format(cfg))

    output_dir = cfg.OUTPUT.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(cfg)
=======
    # cmd_args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='generated_data_fbcampus_2.csv', help='name of the csv file to train on')
    parser.add_argument('--input_size', type=int, default=19, help='input size of the model')
    parser.add_argument('--output_size', type=int, default=3, help='output size of the model')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size to train on')
    args = parser.parse_args()
    cmd_args = vars(args)

    main(cmd_args=cmd_args)
>>>>>>> Stashed changes
