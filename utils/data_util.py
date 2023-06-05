import numpy as np
import pandas as pd
import os
import random
import torch
import matplotlib.pyplot as plt
from torch import nn
# set random seed
random.seed(0)

DATA_DIR = '/home/anees.hashmi/Desktop/ML703/data/2D/evaluation_data'

def get_input_data(seq_len, batch_size, csv_name = None,datadir=DATA_DIR):
    """
    Returns the input data for training the model
    arguments:
        seq_len: length of the sequence
        batch_size: batch size
        datadir: directory where the data is stored
    returns:
        X: input data
        y: labels
        y_kalman: kalman predictions
    """
    # print(os.path.abspath(datadir))
    # CSV_NAME = os.listdir(datadir)[0]
    # print("Loading data from {}".format(os.path.join(datadir, f'{CSV_NAME}')))

    # df = pd.read_csv(os.path.join(datadir, f'{CSV_NAME}'))
    df = pd.read_csv(datadir).dropna()
    df = (df-df.min())/(df.max()-df.min() + 1e-10)
    # df.isna().sum()

    num_batches = (len(df) - seq_len) // (batch_size)

    labels = df[["ground_truth_x", "ground_truth_y", "ground_truth_theta" ]].to_numpy()
    kalman_pred = df[["kalman_prediction_x", "kalman_prediction_y", "kalman_prediction_theta"]].to_numpy() # used for comparison

    if 'ekf_pos_x' in df.columns:
        # df['ekf_pos_theta'] = df['ekf_pos_theta'].apply(lambda x: x*180/np.pi)
        ekf_pred = df[["ekf_pos_x", "ekf_pos_y", "ekf_pos_theta"]].to_numpy() # used for comparison
        ukf_pred = df[["ukf_pos_x", "ukf_pos_y", "ukf_pos_theta"]].to_numpy() # used for comparison


    # df['ukf_pos_theta'] = df['ukf_pos_theta'].apply(lambda x: x*180/np.pi)
    # df['noisy_motion_theta'] = df['noisy_motion_theta'].apply(lambda x: x*180/np.pi)

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
            if 'ekf_pos_x' in df.columns:
                new_ekf_data.append(ekf_pred[i + seq_len -1])
                new_ukf_data.append(ukf_pred[i + seq_len -1])
            # new_ekf_data.append(ekf_pred[i + seq_len -1])
            # new_ukf_data.append(ukf_pred[i + seq_len -1])

    new_input_data = np.array(new_input_data[:num_batches* batch_size]) # drop the last batch
    new_output_data = np.array(new_output_data[:num_batches* batch_size]) # drop the last batch
    new_kp_data = np.array(new_kp_data[:num_batches* batch_size]) # drop the last batch

    if 'ekf_pos_x' in df.columns:
        new_ekf_data = np.array(new_ekf_data[:num_batches* batch_size]) # drop the last batch
        new_ukf_data = np.array(new_ukf_data[:num_batches* batch_size]) # drop the last batch

        return new_input_data, new_output_data, new_kp_data, new_ekf_data, new_ukf_data

    return new_input_data, new_output_data, new_kp_data

def get_input_data_1D(seq_len, batch_size, datadir=DATA_DIR):
    """
    Returns the input data for training the model
    arguments:
        seq_len: length of the sequence
        batch_size: batch size
        datadir: directory where the data is stored
    returns:
        X: input data
        y: labels
        y_kalman: kalman predictions
    """

    # CSV_NAME = os.listdir(datadir)[0]
    # print("Loading data from {}".format(os.path.join(datadir, f'{CSV_NAME}')))

    # df = pd.read_csv(os.path.join(datadir, f'{CSV_NAME}'))
    df = pd.read_csv(datadir).dropna()
    df = (df-df.min())/(df.max()-df.min() + 1e-10)

    num_batches = (len(df) - seq_len) // (batch_size)
    labels = df[["ground truth" ]].to_numpy()
    kalman_pred = df[["kalman prediction"]].to_numpy() # used for comparison

    if 'ekf_pos_x' in df.columns:
        ekf_pred = df[["ekf_pos_x", "ekf_pos_y", "ekf_pos_theta"]].to_numpy() # used for comparison
        ukf_pred = df[["ukf_pos_x", "ukf_pos_y", "ukf_pos_theta"]].to_numpy() # used for comparison


    input_data_df = df[
            ['noisy motion',
            'laser range 1','laser range 2','laser range 3', 'laser range 4', 'laser range 5','laser range 6',
            'motion noise stdev', 'laser noise stdev']
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
            if 'ekf_pos_x' in df.columns:
                new_ekf_data.append(ekf_pred[i + seq_len -1])
                new_ukf_data.append(ukf_pred[i + seq_len -1])
            # new_ekf_data.append(ekf_pred[i + seq_len -1])
            # new_ukf_data.append(ukf_pred[i + seq_len -1])

    new_input_data = np.array(new_input_data[:num_batches* batch_size]) # drop the last batch
    new_output_data = np.array(new_output_data[:num_batches* batch_size]) # drop the last batch
    new_kp_data = np.array(new_kp_data[:num_batches* batch_size]) # drop the last batch

    if 'ekf_pos' in df.columns:
        new_ekf_data = np.array(new_ekf_data[:num_batches* batch_size]) # drop the last batch
        new_ukf_data = np.array(new_ukf_data[:num_batches* batch_size]) # drop the last batch

        return new_input_data, new_output_data, new_kp_data, new_ekf_data, new_ukf_data

    return new_input_data, new_output_data, new_kp_data

def get_data_appended(cfg, config):
    appended_l = []
    if cfg.DATA.SETTING == '2D':
        for i,csv_file in enumerate(os.listdir(cfg.DATA.TRAIN_DATA_DIR)):
            if i == 0:
                appended_l =  get_input_data(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=os.path.join(cfg.DATA.TRAIN_DATA_DIR,csv_file))
                continue
            X =  get_input_data(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=os.path.join(cfg.DATA.TRAIN_DATA_DIR,csv_file))
            appended_l = append(appended_l,X)
    elif cfg.DATA.SETTING == '1D':
        # X =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
        for i,csv_file in enumerate(os.listdir(cfg.DATA.TRAIN_DATA_DIR)):
            if i == 0:
                appended_l =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=os.path.join(cfg.DATA.TRAIN_DATA_DIR,csv_file))
                continue
            X =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=os.path.join(cfg.DATA.TRAIN_DATA_DIR,csv_file))
            appended_l = append(appended_l,X)
    else:
        raise ValueError("Environment not supported")
    return appended_l

def get_data_separate(cfg, config):

    if cfg.DATA.SETTING == '2D':
        X =  get_input_data(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    elif cfg.DATA.SETTING == '1D':
        X =  get_input_data_1D(seq_len = config['sequence_length'], batch_size = cfg.SOLVER.BATCH_SIZE, datadir=cfg.DATA.TRAIN_DATA_DIR)
    else:
        raise ValueError("Environment not supported")
    return X

def train_test_split(data, test_size=0.2):

    '''
    Splits the data into train and test sets
    arguments:
        data: input data of any length example [X, y, y_kalman]
        test_size: size of the test set
    returns:
        train_data: list of train data example [X_train, y_train, y_kalman_train]
        test_data: list of test data[ X_test, y_test, y_kalman_test]
    '''
    data_size = len(data[0])
    
    test_idx = random.sample(range(data_size), k = int(data_size*test_size))
    train_idx = [i for i in range(data_size) if i not in test_idx]

    test_data, train_data = [], []

    for input_data in data:
        # print('')
        # print("input data shape")
        # print(type(input_data))
        # print(input_data.shape)
        # print('')

        test_data.append(input_data[test_idx])
        train_data.append(input_data[train_idx])
    
    return train_data, test_data

def get_dataloader(X,y, batch_size, shuffle=False):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def plot_data(data):
    '''
    TODO: change this to plot the data in a better way
    '''
    plt.figure(figsize=(20, 10))
    size = len(data)
    for i, (key, values) in enumerate(data.items()):

        plt.subplot(size, 1, i+1)

        plt.plot(values)
        plt.plot(data["y_test"])
        
        plt.legend([key, 'y_test'])
        plt.title(key)
        plt.show()

def append(l1,l2):
    appended_l = []
    for i,v in zip(l1,l2):
        appended_l.append(np.concatenate((i,v), axis=0))
    return appended_l

def make_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError
    return optimizer

def make_loss(cfg):

    if cfg.SOLVER.LOSS == 'MSE':
        criterion = nn.MSELoss()
    elif cfg.SOLVER.LOSS == 'L1':
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError

    return criterion

def make_result_dict(cfg):
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
            'GRU': [],
            'LSTM_ln': [],
            'Kalman': [],   
        }
    return results
    
def print_table(cfg, results, models):

    result_df = pd.DataFrame(results)
    final_columns = ['Environment', 'Alpha', 'RNN', 'LSTM', 'GRU', 'LSTM_ln', 'Kalman', 'EKF', 'UKF']

    if cfg.DATA.SETTING == '2D':
        columns = [['environment','alpha','RNN_{i}','LSTM_{i}','GRU_{i}', 'LSTM_ln_{i}','Kalman_{i}','EKF_{i}','UKF_{i}'] for i in ['x','y','theta','total']]

        result_df_x = result_df[columns[0]]
        result_df_y = result_df[columns[1]]
        result_df_theta = result_df[columns[2]]
        result_df_total = result_df[columns[3]]

        result_df_x.columns = final_columns
        result_df_y.columns = final_columns
        result_df_theta.columns = final_columns
        result_df_total.columns = final_columns

        #sort result by environment and alpha

        result_df_x = result_df_x.sort_values(by=['Environment', 'Alpha']).round(3)
        result_df_y = result_df_y.sort_values(by=['Environment', 'Alpha']).round(3)
        result_df_theta = result_df_theta.sort_values(by=['Environment', 'Alpha']).round(3)
        result_df_total = result_df_total.sort_values(by=['Environment', 'Alpha']).round(3)

        # to csv
        result_df_x.to_csv(os.path.join('output','tables', 'result_{}_{}_x.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_y.to_csv(os.path.join('output','tables', 'result_{}_{}_y.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_theta.to_csv(os.path.join('output','tables', 'result_{}_{}_theta.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_total.to_csv(os.path.join('output','tables', 'result_{}_{}_total.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)

        # export to latex
        result_df_x.T.to_latex(os.path.join('output','tables', 'result_{}_{}_x.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=True)
        result_df_y.T.to_latex(os.path.join('output','tables', 'result_{}_{}_y.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=True)
        result_df_theta.T.to_latex(os.path.join('output','tables', 'result_{}_{}_theta.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=True)
        result_df_total.T.to_latex(os.path.join('output','tables', 'result_{}_{}_total.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=True)

    else:

        total_columns = ['Environment','RNN','LSTM','GRU','LSTM_ln','Kalman']

        result_df_total = result_df[total_columns]
        result_df_total = result_df_total.sort_values(by=['Environment']).round(3)

        result_df_total.to_csv(os.path.join('output','tables', 'result_{}_{}_total.csv'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=False)
        result_df_total.T.to_latex(os.path.join('output','tables', 'result_{}_{}_total.tex'.format(cfg.DATA.SETTING, cfg.DATA.SETUP)), index=True)
    
    print(result_df_total.T)

    if cfg.DATA.SETTING == '2D':
        #bar plot results
        bar_plot(cfg, result_df_x, name="X")
        bar_plot(cfg, result_df_y, name="Y")
        bar_plot(cfg, result_df_theta, name="Theta")
        bar_plot(cfg, result_df_total, name="Total")
    #bar plot results
    else:
        bar_plot(cfg, result_df_total, name="Total")

def bar_plot(cfg, df, columns=['RNN','LSTM','GRU','LSTM_ln','Kalman'], name="Total", criterion='MAE'):

    df_plot = df[columns]
    df_plot.index = df['Environment']

    df_plot.plot(kind="bar",figsize=(15, 8))
    plt.title(name)
    plt.ylabel(criterion)
    plt.xlabel('Environment')
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')

    os.makedirs(os.path.join('output','figures'), exist_ok=True)
    plt.savefig(os.path.join('output','figures','{}_{}_{}.png'.format(cfg.DATA.SETTING, cfg.DATA.SETUP, name)))

if __name__ == '__main__':

    datadir = './data/2D/generated_data' 
    # CSV_NAME = os.listdir(datadir)[1]
    # print("Loading data from {}".format(os.path.join(datadir, f'{CSV_NAME}')))

    # X,y, y_kalman = get_input_data(10, 32, datadir=os.path.join(datadir, f'{CSV_NAME}'))

    # print(X.shape)
    appended_l = []
    for i,csv_file in enumerate(os.listdir(datadir)):
        if i == 0:
            appended_l =  get_input_data(seq_len = 10, batch_size = 100, datadir=os.path.join(datadir,csv_file))
            continue
        X =  get_input_data(seq_len = 10, batch_size = 100, datadir=os.path.join(datadir,csv_file))
        appended_l = append(appended_l,X)
        if len([i.shape for i in appended_l if np.isnan(i).any()]):
            print([i.shape for i in appended_l if np.isnan(i).any()])
