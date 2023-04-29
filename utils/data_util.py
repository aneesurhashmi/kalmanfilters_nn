import numpy as np
import pandas as pd
import os
import random
import torch
import matplotlib.pyplot as plt

DATA_DIR = './data/2D/evaluation dataset'

def get_input_data(seq_len, batch_size, datadir=DATA_DIR):
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

    CSV_NAME = os.listdir(datadir)[0]
    print("Loading data from {}".format(os.path.join(datadir, f'{CSV_NAME}')))

    df = pd.read_csv(os.path.join(datadir, f'{CSV_NAME}'))

    num_batches = (len(df) - seq_len) // (batch_size)

    labels = df[["ground_truth_x", "ground_truth_y", "ground_truth_theta" ]].to_numpy()
    kalman_pred = df[["kalman_prediction_x", "kalman_prediction_y", "kalman_prediction_theta"]].to_numpy() # used for comparison

    if 'ekf_pos_x' in df.columns:
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

    CSV_NAME = os.listdir(datadir)[0]
    print("Loading data from {}".format(os.path.join(datadir, f'{CSV_NAME}')))

    df = pd.read_csv(os.path.join(datadir, f'{CSV_NAME}'))

    num_batches = (len(df) - seq_len) // (batch_size)

    labels = df[["Ground truth" ]].to_numpy()
    kalman_pred = df[["Kalman prediction"]].to_numpy() # used for comparison

    if 'ekf_pos_x' in df.columns:
        ekf_pred = df[["ekf_pos_x", "ekf_pos_y", "ekf_pos_theta"]].to_numpy() # used for comparison
        ukf_pred = df[["ukf_pos_x", "ukf_pos_y", "ukf_pos_theta"]].to_numpy() # used for comparison


    input_data_df = df[
            ['noisy_motion','Laser range 6'
            'Laser range 1','Laser range 2','Laser range 3', 'Laser range 4', 'Laser range 5',
            'Motion noise stdev', 'Laser noise stdev']
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

def train_test_split(*data, test_size=0.2):

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