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
import matplotlib.pyplot as plt
import seaborn as sns

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

def save_models(cfg,path, v, i):
     # print('path', path,'\n')
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

def get_configs(cfg):
    configs = {}
    if cfg.DATA.SETUP == 'separated':
        for environemnt in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
            for model in os.listdir(os.path.join(cfg.OUTPUT.OUTPUT_DIR, environemnt)):

                path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, environemnt, model)

                with open(path, 'r') as fp:
                    best_config = json.load(fp)
                if 'LSTM_ln' in model:
                    index = f'{environemnt}/LSTM_ln'
                    print(index)
                    continue
                else:
                    index = f'{environemnt}/{model.split("_")[-1][:-5]}'
                configs[index] = best_config

    elif cfg.DATA.SETUP == 'appended':
        for model in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
            path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, model)
            with open(path, 'r') as fp:
                best_config = json.load(fp)
            index = f'{model.split("_")[-1][:-5]}'
            configs[index] = best_config
    else:
        raise ValueError("Setup not supported")
    return configs

def main(cfg):

    #setup device
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
    print(f'Using device: {device}')


    os.makedirs(os.path.join('output','tables'), exist_ok=True)
    os.makedirs(os.path.join('output','figures'), exist_ok=True)

    configs = get_configs(cfg)

    #setup result dictionary
    if cfg.DATA.SETTING == '2D':
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
        results = {
            'environment': [],
            'LSTM': [],
            'RNN': [],
            # 'GRU': [],
            # 'LSTM_ln': [],
            'Kalman': [],   
        }
    
    #evaluate models
    if cfg.DATA.SETUP == 'separated':
        evaluated_env = []
        for i,v in configs.items():

            cfg.MODEL.TYPE = i.split('/')[1]

            if cfg.DATA.SETTING == '2D':
                cfg.DATA.TRAIN_DATA_DIR = os.path.join(cfg.DATA.EVAL_DATA_DIR, '{}_run_2_merged.csv'.format(i.split('/')[0][15:]))
            else:
                cfg.DATA.TRAIN_DATA_DIR = os.path.join(cfg.DATA.EVAL_DATA_DIR, '{}.csv'.format(i.split('/')[0]))
            
            # save_models(cfg,v, i)
            # continue

            # results['model'] = i.split('/')[1]
            # get data
            test_data = get_data_separate(cfg, v)
            test_data_loader = get_dataloader(test_data[0],test_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)
            if len(test_data_loader)==0:

                print('here')
                continue
            
            # get model
            model, path = get_model(cfg, v)
           
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

                    results['environment'].append(i.split('/')[0][15:-8])
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
                # print(i.split('/')[1], i.split('/')[0])
                results['{}'.format(i.split('/')[1])].append(total_metric)

            # result = pd.DataFrame(results, index=[0])
            # result_df = pd.concat([result_df,result], axis=0)

            evaluated_env.append(i.split('/')[0])
    else:
        evaluated_env = []
        for i,v in configs.items():
            
            # setup model
            cfg.MODEL.TYPE = i
            model,path = get_model(cfg, v)
            for environment in os.listdir(cfg.DATA.EVAL_DATA_DIR):

                cfg.DATA.TRAIN_DATA_DIR = os.path.join(cfg.DATA.EVAL_DATA_DIR, environment)
                # get data
                test_data = get_data_separate(cfg, v)
                test_data_loader = get_dataloader(test_data[0],test_data[1], batch_size=cfg.SOLVER.BATCH_SIZE)

                #infrrence
                loss, test_pred = test_accuracy(model, test_data_loader, device=device)

                # get metric
                each_metric, total_metric = get_all_metric(test_pred, test_data[1], metric='mae')
                if environment not in evaluated_env:
                    each_metric_kf, total_metric_kf = get_all_metric(test_data[2], test_data[1], metric='mae')
                    if cfg.DATA.SETTING == '2D':

                        each_metric_ekf, total_metric_ekf = get_all_metric(test_data[3], test_data[1], metric='mae')
                        each_metric_ukf, total_metric_ukf = get_all_metric(test_data[4], test_data[1], metric='mae')

                        results['environment'].append(environment)
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
                    else:
                        results['environment'].append(environment)
                        results['Kalman'].append(total_metric_kf)
                
                    evaluated_env.append(environment)

            for i in os.listdir()


    if cfg.DATA.SETTING == '2D':
        x_columns = ['environment','alpha','RNN_x','LSTM_x','GRU_x','Kalman_x','EKF_x','UKF_x']
        y_columns = ['environment','alpha','RNN_y','LSTM_y','GRU_y','Kalman_y','EKF_y','UKF_y']
        theta_columns = ['environment','alpha','RNN_theta','LSTM_theta','GRU_theta','Kalman_theta','EKF_theta','UKF_theta']
        total_columns = ['environment','alpha','RNN_total','LSTM_total','GRU_total','Kalman_total','EKF_total','UKF_total']
    else:
        total_columns = ['environment','RNN','LSTM','Kalman']
  
    result_df = pd.DataFrame(results)

    if cfg.DATA.SETTING == '2D':

        result_df_x = result_df[x_columns]
        result_df_y = result_df[y_columns]
        result_df_theta = result_df[theta_columns]
        result_df_total = result_df[total_columns]

        result_df_x.columns = ['Environment','Alpha','RNN','LSTM','GRU','KF','EKF','UKF']
        result_df_y.columns = ['Environment','Alpha','RNN','LSTM','GRU','KF','EKF','UKF']
        result_df_theta.columns = ['Environment','Alpha','RNN','LSTM','GRU','KF','EKF','UKF']
        result_df_total.columns = ['Environment','Alpha','RNN','LSTM','GRU','KF','EKF','UKF']

        result_df_x = result_df_x.round(5)
        result_df_y = result_df_y.round(5)
        result_df_theta = result_df_theta.round(5)
        result_df_total = result_df_total.round(5)

        result_df_x.to_csv(os.path.join('output','tables', 'result_{}_{}_x.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_y.to_csv(os.path.join('output','tables', 'result_{}_{}_y.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_theta.to_csv(os.path.join('output','tables', 'result_{}_{}_theta.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_total.to_csv(os.path.join('output','tables', 'result_{}_{}_total.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)

        # export to latex
        result_df_x.to_latex(os.path.join('output','tables', 'result_{}_{}_x.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_y.to_latex(os.path.join('output','tables', 'result_{}_{}_y.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_theta.to_latex(os.path.join('output','tables', 'result_{}_{}_theta.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_total.to_latex(os.path.join('output','tables', 'result_{}_{}_total.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)

    else:

        result_df_total = result_df[total_columns]
        result_df_total.columns = ['Environment','RNN','LSTM','KF']
        result_df_total = result_df_total.round(5)

        result_df_total.to_csv(os.path.join('output','tables', 'result_{}_{}.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)

        #export to latex
        result_df_total.to_latex(os.path.join('output','tables', 'result_{}_{}.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)

    #bar plot results
    bar_plot(cfg, result_df_total)

    print('Done')

def bar_plot(cfg, df):
    # if cfg.DATA.SETTING == '2D':
    #     df_plot = df[['RNN','LSTM','GRU','KF','EKF','UKF']]
    #     df_plot.index = df['Environment']

    #     df_plot.plot(kind="bar",figsize=(15, 8))
    #     plt.title('Total MAE')
    #     plt.ylabel('MAE')
    #     plt.xlabel('Environment')
    #     plt.xticks(rotation=0)
    #     plt.legend(loc='upper right')

    #     os.makedirs(os.path.join('output','figures'), exist_ok=True)
    #     plt.savefig(os.path.join('output','figures','{}_{}.png'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)))
    df_plot = df[['RNN','LSTM','KF',]]
    df_plot.index = df['Environment']

    df_plot.plot(kind="bar",figsize=(15, 8))
    plt.title('Total MAE')
    plt.ylabel('MAE')
    plt.xlabel('Environment')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')

    os.makedirs(os.path.join('output','figures'), exist_ok=True)
    plt.savefig(os.path.join('output','figures','{}_{}.png'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)))

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
    # cfg.SOLVER.BATCH_SIZE = 50
    # cfg.freeze()



    main(cfg)

