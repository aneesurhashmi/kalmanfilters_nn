import rospy 
from std_msgs.msg import Float64 
import sys 
import numpy as np
from ml703.msg import pose2d
from geometry_msgs.msg import PoseWithCovariance
# initial estimates of position 

def state_update1D(mu_t, mu_motion, sig_t, sig_motion):
    mu_new = mu_t + mu_motion
    sig_new = sig_t + sig_motion 
    return mu_new, sig_new 

def correction_step1D(mu_x, mu_z, sig_x, sig_z):
    mu_new = ((mu_x * sig_z) + (mu_z * sig_x)) / (sig_x + sig_z)
    sig_new = (sig_x * sig_z) / (sig_x + sig_z) 
    return mu_new, sig_new 

def laser_callback(msg):
    global dim
    if dim == '1':
        global cur_measurement
        cur_measurement = msg.data  
    else:
        global mu_z, cov_z 
        mu_z = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.z])
        temp = np.array(msg.covariance).reshape((6, 6))
        # rospy.loginfo(temp)
        cov_z = np.array([[temp[0][0], temp[0][1], temp[0][5]],
                          [temp[1][0], temp[1][1], temp[1][5]],
                          [temp[5][0], temp[5][1], temp[5][5]]
                ])

def odom_callback(msg):
    # We have received motion updates. So we need to update the state of the robot 
    global belief_pub, dim
    if dim == '1':
        global cur_odom, cur_measurement, motion_sig, last_odom
        global measurement_sig, mu, sig
        
        cur_odom = msg.data - last_odom
        mu, sig = state_update1D(mu, cur_odom, sig, motion_sig)
        rospy.loginfo(f'Prediction: [{mu}, {sig}]')
        mu, sig = correction_step1D(mu, cur_measurement, sig, measurement_sig)
        rospy.loginfo(f'Correction: [{mu}, {sig}]')
        last_odom = msg.data 
        belief_pub.publish(mu)
    else:
        global mu_m, mu_z, cov_m, cov_z 
        mu_m = mu_m + np.array([msg.x, msg.y, msg.theta]) 
        cov_m = np.array(msg.covariance).reshape((3,3))
        kalman_gain = cov_m @ np.linalg.inv(cov_m + cov_z)
        # rospy.loginfo(mu_m)
        mu_m = mu_m + kalman_gain @ (mu_z - mu_m)
        
        cov_m -= kalman_gain @ cov_m
        belief = pose2d()
        belief.x, belief.y, belief.theta = mu_m[0], mu_m[1], mu_m[2]
        belief.covariance = cov_m.flatten()
        # rospy.loginfo(belief)
        belief_pub.publish(belief)

if __name__ == '__main__':
    cur_odom = None
    cur_measurement = None 
    motion_sig = 4.0
    last_odom = 0.0
    measurement_sig = 0.01
    dim = sys.argv[2]
    rospy.loginfo('Starting kalman filter')
    rospy.init_node('kalman_filter', log_level=rospy.INFO)
    if (dim == '1'):
        mu = 0.0
        sig = 1000 
        # get noisy odometry
        belief_pub = rospy.Publisher('/kalman_belief', Float64, queue_size=1)
        laser_sub = rospy.Subscriber('/turtlebot3_pose', Float64, laser_callback)
    else:
        mu_m = np.full((3,), 0)
        cov_m = np.ones((3,3))
        mu_z = np.full((3,), 0)
        cov_z = np.zeros((3, 3))
        odom_sub = rospy.Subscriber('/noisy_odom', pose2d, odom_callback)
        laser_sub = rospy.Subscriber('/pose_with_covariance_lidar', PoseWithCovariance, laser_callback)
        belief_pub = rospy.Publisher('/kalman_belief', pose2d, queue_size=1)

    rospy.spin()