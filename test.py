import json
import ray
from ray import tune
from model import RNN,  Base
from utils import plot_data, get_input_data, get_dataloader, get_input_data_1D
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from config import cfg
import argparse
from model import Base
import shutil

def test_accuracy(net, test_loader, criterion=nn.MSELoss(), device="cpu"):
    
    net.to(device)
    loss = 0
    total = 0
    test_pred = []
    with torch.no_grad():
        for data in test_loader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            outputs, _ = net(input)
            # _, predicted = torch.max(outputs.data, 1)
            total += input.size(0)
            loss += criterion(outputs, labels).cpu().numpy()
            test_pred.append(outputs.cpu().numpy())
    return None , np.concatenate(np.array(test_pred), axis=0)

def get_loss(pred, labels, metric = "mae"):
    if metric == "mae":
        criterion = nn.L1Loss()
    elif metric == "mse":
        criterion = nn.MSELoss()
    return criterion(torch.tensor(pred).clone().detach() , torch.tensor(labels)).item()

def get_all_metric(pred, labels, metric):
    all_losses = []
    for i in range(pred.shape[1]): # output size 
        all_losses.append(get_loss(pred[:, i], labels[:, i], metric=metric))
    return all_losses, get_loss(pred[:, 0], labels[:, 0]) # return mse for x, y, theta and total mse

def get_model(cfg, config):
    model = Base(input_size=cfg.MODEL.INPUT_SIZE,
                hidden_size = config["hidden_size"],
                num_layers = config["num_layers"],
                output_size=1 if cfg.DATA.SETTING == '1D' else 3,
                model = cfg.MODEL.TYPE,
                )
    
    if os.path.isdir(config['dir']):
        chkpoints_list = [i for i in os.listdir(config['dir']) if i.startswith("checkpoint")]
        last_checkpoint = sorted(chkpoints_list, key=lambda x: int(x.split('_')[-1]))[-1]
        chkpoint_state_dict, optim = torch.load(os.path.join(config['dir'], last_checkpoint, 'checkpoint'))

        path = os.path.join(config['dir'], last_checkpoint, 'checkpoint')

    else:
        chkpoint_state_dict = torch.load(os.path.join(config['dir']))
        path = os.path.join(config['dir'])
    model.load_state_dict(chkpoint_state_dict)

    return model, path

def get_data_separate(cfg, config):

    if cfg.DATA.SETTING == '2D':
        X =  get_input_data(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    elif cfg.DATA.SETTING == '1D':
        X =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    else:
        raise ValueError("Environment not supported")
    return X

def main(cfg):


    #setup device
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f'Using device: {device}')


    configs = {}
    for environemnt in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
        for model in os.listdir(os.path.join(cfg.OUTPUT.OUTPUT_DIR, environemnt)):

            path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, environemnt, model)

            with open(path, 'r') as fp:
                best_config = json.load(fp)
            if 'LSTM_ln' in model:
                index = f'{environemnt}/LSTM_ln'
                print(index)
            else:
                index = f'{environemnt}/{model.split("_")[-1][:-5]}'
            configs[index] = best_config
    if cfg.DATA.SETTING == '2D':
        result_df = pd.DataFrame(columns=['model','enviroment','mse_x','mse_y','mse_theta','mse_total',
                                          'mse_x_kf','mse_y_kf','mse_theta_kf','mse_total_kf',
                                          'mse_x_ekf','mse_y_ekf','mse_theta_ekf','mse_total_ekf',
                                          'mse_x_ukf','mse_y_ukf','mse_theta_ukf','mse_total_ukf', 'alpha'])
        results = {
            'environment': [], 
            'LSTM_x': [], 'LSTM_y': [], 'LSTM_theta': [], 'LSTM_total': [],
            'RNN_x': [], 'RNN_y': [], 'RNN_theta': [], 'RNN_total': [],
            'GRU_x': [], 'GRU_y': [], 'GRU_theta': [], 'GRU_total': [],
            'LSTM_ln_x': [], 'LSTM_ln_y': [], 'LSTM_ln_theta': [], 'LSTM_ln_total': [],
            'Kalman_x': [], 'Kalman_y': [], 'Kalman_theta': [], 'Kalman_total': [],
            'EKF_x': [], 'EKF_y': [], 'EKF_theta': [], 'EKF_total': [],
            'UKF_x': [], 'UKF_y': [], 'UKF_theta': [], 'UKF_total': [],
            'alpha': []
        }
    else:
        result_df = pd.DataFrame(columns=['model','enviroment','mse_total','mse_total_kf'])
    
        results = {
            'environment': [],
            'LSTM': [],
            'RNN': [],
            'GRU': [],
            'LSTM_ln': [],
            'Kalman': [],
            
        }
    evaluated_env = []
    for i,v in configs.items():

        cfg.MODEL.TYPE = i.split('/')[1]

        if cfg.DATA.SETTING == '2D':
            cfg.DATA.TRAIN_DATA_DIR = os.path.join(cfg.DATA.EVAL_DATA_DIR, '{}_run_2_merged.csv'.format(i.split('/')[0][15:]))
        else:
            cfg.DATA.TRAIN_DATA_DIR = os.path.join(cfg.DATA.EVAL_DATA_DIR, '{}.csv'.format(i.split('/')[0]))
        
        # results['model'] = i.split('/')[1]
        
        

        

        # get data
        test_data = get_data_separate(cfg, v)
        test_data_loader = get_dataloader(test_data[0],test_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)
        if len(test_data_loader)==0:
            continue
        # get model
        model, path = get_model(cfg, v)
       
        print(path)
        if cfg.DATA.SETTING == '2D':
            new_path = os.path.join('output', '{}_{}_{}_{}.pth'.format(cfg.DATA.SETTING,i.split('/')[0][15:], cfg.MODEL.TYPE, cfg.DATA.SETUP))
            v['dir'] = new_path
            shutil.copy(path, new_path )
            with open('{}/best_config_{}.json'.format('output', '{}_{}_{}_{}'.format(cfg.DATA.SETTING,i.split('/')[0][15:], cfg.MODEL.TYPE, cfg.DATA.SETUP)), 'w') as fp:
                json.dump(v, fp, sort_keys=True, indent=4)
        else:
            new_path = os.path.join('output', '{}_{}_{}_{}.pth'.format(cfg.DATA.SETTING,i.split('/')[0], cfg.MODEL.TYPE, cfg.DATA.SETUP))
            v['dir'] = new_path
            shutil.copy(path, new_path)
            with open('{}/best_config_{}.json'.format('output', '{}_{}_{}_{}'.format(cfg.DATA.SETTING, i.split('/')[0], cfg.MODEL.TYPE, cfg.DATA.SETUP)), 'w') as fp:
                json.dump(v, fp, sort_keys=True, indent=4)
        
        # inference
        loss, test_pred = test_accuracy(model, test_data_loader, device=device)

        # get metric
        each_metric, total_metric = get_all_metric(test_pred, test_data[1], metric='mae')

        if i.split('/')[0] not in evaluated_env:
            each_metric_kf, total_metric_kf = get_all_metric(test_data[2], test_data[1], metric='mae')

            if cfg.DATA.SETTING == '2D':
                results['alpha'].append( i.split('/')[0].split('_')[-1])

            if cfg.DATA.SETTING == '2D':
                each_metric_ekf, total_metric_ekf = get_all_metric(test_data[3], test_data[1], metric='mse')
                each_metric_ukf, total_metric_ukf = get_all_metric(test_data[4], test_data[1], metric='mse')

                results['environment'].append(i.split('/')[0])
                results['EKF_x'].append(each_metric_ekf[0])
                results['EKF_y'].append(each_metric_ekf[1])
                results['EKF_theta'].append(each_metric_ekf[2])
                results['EKF_total'].append(total_metric_ekf)
                
                results['UKF_x'].append(each_metric_ukf[0])
                results['UKF_y'].append(each_metric_ukf[1])
                results['UKF_theta'].append(each_metric_ukf[2])
                results['UKF_total'].append(total_metric_ukf)

                
                results['Kalman_x'].append(each_metric_kf[0])
                results['Kalman_y'].append(each_metric_kf[1])
                results['Kalman_theta'].append(each_metric_kf[2])
                results['Kalman_total'].append(total_metric_kf)
            

            
            elif cfg.DATA.SETTING == '1D':
                results['environment'].append(i.split('/')[0])
                results['Kalman'].append(total_metric_kf)
        if cfg.DATA.SETTING == '2D':
            results['{}_x'.format(i.split('/')[1])].append(each_metric[0])
            results['{}_y'.format(i.split('/')[1])].append(each_metric[1])
            results['{}_theta'.format(i.split('/')[1])].append(each_metric[2])
            results['{}_total'.format(i.split('/')[1])].append(total_metric)
        else:
            results['{}'.format(i.split('/')[1])].append(total_metric)

        # result = pd.DataFrame(results, index=[0])
        # result_df = pd.concat([result_df,result], axis=0)

        evaluated_env.append(i.split('/')[0])
    
    # result_df_x = pd.DataFrame([result_df['model'],result_df['enviroment'],result_df['mse_x'],result_df['mse_x_kf'],result_df['mse_x_ekf'],result_df['mse_x_ukf']]).T
    # results = pd.DataFrame(results)
    # results.to_csv(os.path.join(cfg.OUTPUT.OUTPUT_DIR, 'result.csv'), index=False)

    # export to latex table
    # if cfg.DATA.SETTING == '2D':
    #     result_df_total = result_df[['model','enviroment','mse_total','mse_total_kf','mse_total_ekf','mse_total_ukf']]
    #     result_df_total.columns = ['Model','Environment','NN','KF','EKF','UKF']
    #     result_df_total = result_df_total.round(3)
    #     result_df_total.to_latex(os.path.join(cfg.OUTPUT.OUTPUT_DIR, 'result_total.tex'), index=False)

    #     result_df_x = result_df[['model','enviroment','mse_x','mse_x_kf','mse_x_ekf','mse_x_ukf']]
    #     result_df_x.columns = ['Model','Environment','NN','KF','EKF','UKF']
    #     result_df.columns = ['Model','Environment','NN','KF','EKF','UKF']
    # else:
    #     result_df = result_df[['model','enviroment','mse_total','mse_total_kf']]
    #     result_df.columns = ['Model','Environment','NN','KF']
    # result_df = result_df.round(3)
    # result_df.to_latex(os.path.join(cfg.OUTPUT.OUTPUT_DIR, 'result.tex'), index=False)
    # print(pd.DataFrame.from_dict(results))
    # print(results)
    # output_path = cfg.OUTPUT.OUTPUT_DIR
    # with open('{}/test_results.json'.format(output_path), 'w') as fp:
    #     json.dump(results, fp, sort_keys=True, indent=4)

    print('Done')


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
    cfg.SOLVER.BATCH_SIZE = 50
    # cfg.freeze()

    main(cfg)

