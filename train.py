import torch
from torch import nn
from model import Base, make_model
from torch import optim
import os
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray
from ray import tune, air
from utils import *
import numpy as np
import json
import random
import argparse
from config import cfg
from tqdm import tqdm

random.seed(0)
torch.manual_seed(0)

def train_ray(config,cfg):
    device="cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    print("current working directory: {}".format(os.getcwd()))
    # get data
    if cfg.DATA.SETUP == 'appended':
        appended_l = get_data_appended(cfg, config)
    elif cfg.DATA.SETUP == 'separated':
        appended_l = get_data_separate(cfg, config)
    else:
        raise ValueError("Setup not supported")
    
    train_data, valid_data = train_test_split(appended_l, test_size=cfg.DATA.TEST_SIZE)

    train_loader = get_dataloader(train_data[0],train_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)
    valid_loader = get_dataloader(valid_data[0],valid_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)

    print("Using model: {}".format(cfg.MODEL.TYPE)) # setup model
    model, _ = make_model(cfg)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = make_loss(cfg)
    optimizer = make_optimizer(cfg, model)

    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, hidden = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
                loss = criterion(outputs, labels)
                
                val_loss += loss.cpu().numpy()
                total += labels.size(0)
                val_steps += 1
                
        if epoch % cfg.SOLVER.LOG_STEP == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))

    print("Finished Training")

def test_accuracy(net, test_loader,cfg, device="cpu"):
    
    net.to(device)
    criterion = make_loss(cfg)
    loss = 0
    total = 0
    test_pred = []
    with torch.no_grad():
        for data in test_loader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            outputs, _ = net(input)

            total += labels.size(0)
            loss += criterion(outputs, labels).cpu().numpy()
            test_pred.append(outputs.cpu().numpy())

    return loss / total, test_pred

def main(cfg):
    gpus_per_trial = cfg.SOLVER.GPUS_PER_TRIAL
    # configurable parameters
    config = {
        "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_layers": tune.choice([4, 12, 32, 64]),
        'sequence_length': tune.choice([14, 28, 38, 56]),
    }

    # train using raytune
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=cfg.SOLVER.NUM_EPOCHS,
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
    if best_trial is not None:
        best_trial.config['dir'] = str(best_trial.log_dir)
        best_trial.config[cfg.SOLVER.LOSS] = best_trial.metrics["loss"]
    else:
        raise ValueError("No trials found ")
       
    if cfg.DATA.SETUP == 'appended':  # save best config as json
        output_path = cfg.OUTPUT.OUTPUT_DIR
    else:
        output_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, cfg.DATA.TRAIN_DATA_DIR.split('/')[-1][:-4])

    os.makedirs(output_path, exist_ok=True)

    with open('{}/best_config_{}.json'.format(output_path, cfg.MODEL.TYPE), 'w') as fp:
        json.dump(best_trial.config, fp, sort_keys=True, indent=4)

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))

# -------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- MAIN ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
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
    # train for 100 epochs more
    # get data
    for model in os.listdir(cfg.OUTPUT.OUTPUT_DIR):

        path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, model)
        with open(path, 'r') as fp:
            best_config = json.load(fp)
        
        train_best_net(best_config, cfg, checkpoint=True)