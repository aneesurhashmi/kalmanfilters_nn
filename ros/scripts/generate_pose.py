import rospy 
from geometry_msgs.msg import Twist, TwistWithCovarianceStamped, TwistWithCovariance, PoseWithCovariance, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry 
import numpy as np

def odom_callback(msg):
    global header
    header = msg.header 

def twist_callback(msg):
    global header, twist_pub
    if header is None:
        return 
    new_msg = TwistWithCovarianceStamped()
    new_msg.header = header 
    new_msg.twist = TwistWithCovariance()
    new_msg.twist.twist = msg 
    new_msg.twist.covariance = np.zeros((36,))
    twist_pub.publish(new_msg)

def pose_callback(msg):
    global header, pose_pub
    if header is None: 
        return
    new_msg = PoseWithCovarianceStamped()
    new_msg.header = header
    new_msg.pose = msg
    pose_pub.publish(new_msg)

if __name__ == '__main__':
    rospy.init_node('generate_pose', log_level=rospy.INFO)
    header = None 
    odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback)
    cmd_sub = rospy.Subscriber('/cmd_vel', Twist, twist_callback)
    pose_sub = rospy.Subscriber('/pose_with_covariance_lidar', PoseWithCovariance, pose_callback)
    
    pose_pub = rospy.Publisher('/pose_stamped', PoseWithCovarianceStamped, queue_size=1)
    # publisher = rospy.Publisher('/noisy_odom_stamped', Odometry, odom_callback)
    twist_pub = rospy.Publisher('/twist_stamped', TwistWithCovarianceStamped, queue_size=1)
    rospy.spin()