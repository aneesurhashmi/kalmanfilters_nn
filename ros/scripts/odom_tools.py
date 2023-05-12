import rospy 
import numpy as np 
from std_msgs.msg import Float64 
from ml703.msg import pose2d 
from nav_msgs.msg import Odometry 
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist 
import sys 

def sim1d():
    global cur_odom, total_odom, noisy_odom, pub, truth_pub
    cur_odom = np.random.normal(0.5, sd_odom_sample)
    total_odom += cur_odom
    
    update_state(total_odom)
    # request teleport service if x > 8.1
    if total_odom > 9.5 or total_odom < -20:
        total_odom = 0.0
        update_state(total_odom)
        
    sd_odom = alpha * cur_odom
    noisy_odom += np.random.normal(cur_odom, sd_odom**2)
    rospy.loginfo(f'Publishing {noisy_odom}')
    pub.publish(noisy_odom)
    truth_pub.publish(total_odom)

def sim2d(state_msg):
    global cur_odom, noisy_odom
    # sample from multivariate gaussian with cur_odom as mean and
    x = state_msg.pose.pose.position.x 
    y = state_msg.pose.pose.position.y 
    theta = state_msg.pose.pose.orientation.z
    # rospy.loginfo(f'truth: {x, y, theta}')
    pos = np.array([x, y, theta])
    delta = pos - cur_odom
    # rospy.loginfo(delta)
    cur_odom = pos
    # rospy.loginfo(f'delta: {delta}')
    sample_cov = np.outer((alpha * delta), (alpha * delta))
    # rospy.loginfo(f'sample cov: {sample_cov}')
    cur_cov = np.multiply(sample_cov, cov_odom)
    x_hat, y_hat, theta_hat = np.random.multivariate_normal(delta, cur_cov)
    noisy_odom = [x_hat, y_hat, theta_hat]
    msg = pose2d()
    msg.x = x_hat 
    msg.y = y_hat 
    msg.theta = theta_hat
    msg.covariance = cur_cov.flatten()
    noisy_odom = np.array(noisy_odom)
    # msg.data = [x_hat, y_hat, theta_hat]
    rospy.loginfo(f'Publishing {noisy_odom}')
    pub.publish(msg) 

def update_state(pos_x):
    global teleport_pub 
    state_msg = ModelState()
    state_msg.model_name = 'turtlebot3_burger'
    state_msg.reference_frame = 'ground_plane'
    state_msg.pose.position.x = pos_x
    state_msg.pose.position.y = 0.0
    state_msg.pose.position.z = 0.0
    state_msg.pose.orientation.x = 0.0
    state_msg.pose.orientation.y = 0.0
    state_msg.pose.orientation.z = 0.0
    state_msg.pose.orientation.w = 1.0
    teleport_pub.publish(state_msg)

if __name__ == '__main__':
    rospy.init_node('odom_tools')
    dim = sys.argv[2]
    print(dim)
    if dim == '1':
        cur_odom = None
        noisy_odom = 0.0
        alpha = 8
        sd_odom_sample = 0.4
        total_odom = 0.0
        pub = rospy.Publisher('/noisy_odom', Float64, queue_size=1)
        cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    elif dim == '2':
        cur_odom = np.zeros((3,))
        alpha = 8
        noisy_odom = np.zeros((3,))
        cov_odom = np.full((3,3), 1)
        pub = rospy.Publisher('/noisy_odom', pose2d, queue_size=1)
        cmd_sub = rospy.Subscriber('/odom', Odometry, sim2d)
        
    
    # publish control commands, adds gaussian noise, returns 
    # sub = rospy.Subscriber('/odom', Odometry, odom_callback)
    
    truth_pub = rospy.Publisher('/odom_truth', Float64, queue_size=1)
    teleport_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

    r = rospy.Rate(5)
    while ~rospy.is_shutdown():
        # get random motion vector
        if dim == 1:
            sim1d()
        r.sleep()
    rospy.spin()