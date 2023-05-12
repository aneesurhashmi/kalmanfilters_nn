import rospy
from sensor_msgs.msg import LaserScan 
from std_msgs.msg import Float64
import numpy as np 

def laser_callback(msg):
    global last_scan, pub, init_pose, sig 
    # get the min of the front 12 laser scans
    min_range = min(min(msg.ranges[354:363]), 12)
    # add noise to measurement
    # min_range = np.random.normal(min_range, sig)
    # compute distance as original distance minus current scan
    x_pos = init_pose - min_range 
    # publish position
    rospy.loginfo(f'Publishing: {x_pos}')
    pub.publish(x_pos)

if __name__ == '__main__':
    rospy.init_node('laser_scanner')
    last_scan = None
    if rospy.has_param('init_pose'):
        init_pose = rospy.get_param('init_pose')
    else:
        rospy.logwarn('No initial position for obstacle provided. Using default')
        init_pose = 10.0
    if rospy.has_param('sig'):
        sig = rospy.get_param('sig')
    else:
        rospy.logwarn('Standard deviation for sensor measurements not provided. Using default')
    sig = 0.1
    pub = rospy.Publisher('/turtlebot3_pose', Float64, queue_size=1)
    sub = rospy.Subscriber('/scan', LaserScan, laser_callback)
    
    rospy.spin()