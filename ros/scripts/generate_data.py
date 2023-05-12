import rosbag 
import pandas as pd 
import numpy as np

bag = rosbag.Bag('./Bags/KF/fbcampus_alpha_8.bag')
# topics = ['/noisy_odom', '/scan', '/odom_truth', '/kalman_belief']
topics = ['/noisy_odom', '/pose_with_covariance_lidar', '/odom', '/kalman_belief']
# topics = ['/odometry/filtered_throttled']

column_names = ['noisy_motion_x', 'noisy_motion_y', 'noisy_motion_theta',
                'noisy_motion_cov_xx', 'noisy_motion_cov_xy', 'noisy_motion_cov_xtheta',
                'noisy_motion_cov_yy', 'noisy_motion_cov_ytheta', 'noisy_motion_cov_thetatheta', 
                'kalman_prediction_x', 'kalman_prediction_y', 'kalman_prediction_theta',
                'lidar_x', 'lidar_y', 'lidar_theta', 'lidar_cov_xx', 'lidar_cov_xy', 'lidar_cov_xtheta',
                'lidar_cov_yy', 'lidar_cov_ytheta', 'lidar_cov_thetatheta',
                'ground_truth_x', 'ground_truth_y', 'ground_truth_theta']
# column_names = ['ukf_pos_x', 'ukf_pos_y', 'ukf_pos_theta']

# for i in range(1, 13):
#     column_names.append(f'laser range {i}')

df = pd.DataFrame(columns=column_names)

def read_message(topic, msg, row):
    # global df 
    # df.loc[row, 'ukf_pos_x'] = msg.pose.pose.position.x
    # df.loc[row, 'ukf_pos_y'] = msg.pose.pose.position.y
    # df.loc[row, 'ukf_pos_theta'] = msg.pose.pose.orientation.z
    
    if topic == '/noisy_odom':
        df.loc[row, 'noisy_motion_x'] = msg.x
        df.loc[row, 'noisy_motion_y'] = msg.y
        df.loc[row, 'noisy_motion_theta'] = msg.theta
        df.loc[row, 'noisy_motion_cov_xx'] = msg.covariance[0]
        df.loc[row, 'noisy_motion_cov_xy'] = msg.covariance[1]
        df.loc[row, 'noisy_motion_cov_xtheta'] = msg.covariance[2]
        df.loc[row, 'noisy_motion_cov_yy'] = msg.covariance[4]
        df.loc[row, 'noisy_motion_cov_ytheta'] = msg.covariance[5]
        df.loc[row, 'noisy_motion_cov_thetatheta'] = msg.covariance[8]

    elif topic == '/kalman_belief':
        df.loc[row, 'kalman_prediction_x'] = msg.x
        df.loc[row, 'kalman_prediction_y'] = msg.y
        df.loc[row, 'kalman_prediction_theta'] = msg.theta

    elif topic == '/pose_with_covariance_lidar':
        # ranges = np.array(msg.ranges[354:363])
        # for i in range(len(ranges)):
        #     df.loc[row, f'laser range {i+1}'] = ranges[i]
        df.loc[row, 'lidar_x'] = msg.pose.position.x
        df.loc[row, 'lidar_y'] = msg.pose.position.y
        df.loc[row, 'lidar_theta'] = msg.pose.orientation.z
        df.loc[row, 'lidar_cov_xx'] = msg.covariance[0]
        df.loc[row, 'lidar_cov_xy'] = msg.covariance[1]
        df.loc[row, 'lidar_cov_xtheta'] = msg.covariance[5]
        df.loc[row, 'lidar_cov_yy'] = msg.covariance[7]
        df.loc[row, 'lidar_cov_ytheta'] = msg.covariance[11]
        df.loc[row, 'lidar_cov_thetatheta'] = msg.covariance[35]


    else:
        df.loc[row, 'ground_truth_x'] = msg.pose.pose.position.x
        df.loc[row, 'ground_truth_y'] = msg.pose.pose.position.y
        df.loc[row, 'ground_truth_theta'] = msg.pose.pose.orientation.z

  
for topic in topics:
    row = 0
    for topic, msg, t in bag.read_messages(topics=topic):  
        read_message(topic, msg, row)
        row += 1

df['alpha'] = 8
df.to_csv('fbcampus_alpha_8_fixed.csv', index=False)