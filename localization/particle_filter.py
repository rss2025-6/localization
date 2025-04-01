from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion

from rclpy.node import Node
import rclpy

import numpy as np
assert rclpy

pi = np.pi

sin = np.sin
cos = np.cos
atan2 = np.arctan2

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        self.particles = None
        self.likelihood_table = None
        # self.new_particles = None
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    # Determine the "average" (term used loosely) particle pose and publish that transform.
    def averager(self):

        avg_pose = np.average(self.particles[:,:2], axis=0, weights=self.likelihood_table)

        # get sum of sins & cos of angles to find avg theta
        thetas = self.particles[:,2]
        s_thetas = np.sum(sin(thetas))
        c_thetas = np.sum(cos(thetas))

        odom = Odometry()

        odom.pose.pose.position.x = avg_pose[0]
        odom.pose.pose.position.y = avg_pose[1]
        odom.pose.pose.orientation = atan2(s_thetas, c_thetas)

        self.odom_pub.publish(odom)

    # Whenever you get odometry data use the motion model to update the particle positions
    def odom_callback(self, msg):
        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y

        # orientation = msg.pose.pose.orientation

        # roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        dtheta = msg.twist.twist.angular.z
        if self.particles:
            # self.motion_model.evaluate(self.particles, [x, y, yaw])
            self.particles = self.motion_model.evaluate(self.particles, [dx, dy, dtheta])
        
        self.averager()

    # Whenever you get sensor data use the sensor model to compute the particle probabilities. 
    # Then resample the particles based on these probabilities
    def laser_callback(self, msg):
        if self.particles:
            self.likelihood_table = self.sensor_model.evaluate(self.particles, msg.ranges)

        self.averager()

    # initial pose
    def pose_callback(self, msg):

        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y

        # orientation = msg.pose.pose.orientation

        # roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        odom = Odometry()

        # odom.pose.pose.position.x = x
        # odom.pose.pose.position.y = y
        odom.pose.pose.position = msg.pose.pose.position
        odom.pose.pose.orientation = msg.pose.pose.orientation

        self.odom_pub.publish(odom)

        # Generate random distribution around x, y
        num_particles = 200
        resamples = np.random.choice(self.particles, num_particles, self.likelihood_table)
        
        # x_samples = np.random.uniform(-2, 2, num_particles) + msg.pose.pose.position.x
        # y_samples = np.random.uniform(-2, 2, num_particles) + msg.pose.pose.position.y
        # theta_samples = np.random.uniform(-pi, pi, num_particles)

        # self.particles = np.hstack((x_samples.T, y_samples.T, theta_samples.T))
        self.particles = np.hstack(resamples)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
