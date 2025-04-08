from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, TransformStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
import rclpy
from transforms3d.quaternions import quat2mat, mat2quat
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

        # Initialize transform listener
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize transform listener
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('pow_value', 1)
        #self.declare_parameter('num_beams_per_particle', 200)
        #self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.pow_value = self.get_parameter("pow_value").get_parameter_value().double_value

        self.tf_broadcaster = TransformBroadcaster(self)
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

        # Initialize publisher for particles array
        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)

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

        # Initialize particles, likelihood table, and number of particles
        self.particles = None
        self.likelihood_table = None
        self.num_particles = 200

        # Initialize variables so the laser update is published less freqently than odom
        self.laser_ct = 0
        self.laser_rate = 50

        # Initialize previous time for odom update
        self.prev_time = self.get_clock().now().nanoseconds

        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    # create T matrix based on position [[x],[y],[z]] and orientation [x,y,z,w]
    def pose2arr(self, p, q):

        # Find rotation matrix, R, from quaternion
        R = quat2mat(q)

        # create T
        T = np.hstack((R, p))  
        T = np.vstack((T, [0,0,0,1]))

        return T
    
    def sendBroadcast(self, header_frame, child_frame, T):

        # initialize transform
        tf = TransformStamped()

        # setup header
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = header_frame

        # set child
        tf.child_frame_id = child_frame

        # get position & orientation data from T
        p = T[:3, 3]
        q = mat2quat(T[:3, :3])

        # set position
        tf.transform.translation.x = p[0]
        tf.transform.translation.y = p[1]
        tf.transform.translation.z = p[2]

        # set orientation
        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]

        # broadcast transform
        self.tf_broadcaster.sendTransform(tf)


    # Determine the "average" (term used loosely) particle pose and publish that transform.
    def averager(self):

        # Get weigthed average position of all the particles
        #avg_pose = np.average(self.particles[:,:2], axis=0, weights=self.likelihood_table)
        avg_pose = np.average(self.particles[:, :2], axis = 0)

        # Get weighted sum of sins & cos of angles to find average theta
        thetas = self.particles[:,2]
        #s_thetas = np.average(sin(thetas), axis=0, weights=self.likelihood_table)
        #c_thetas = np.average(cos(thetas), axis=0, weights=self.likelihood_table)
        s_thetas = np.average(sin(thetas), axis=0)
        c_thetas = np.average(cos(thetas), axis=0)

        # Generate odometry message
        odom = Odometry()

        # Set frame to map
        odom.header.frame_id = "/map"
        odom.child_frame_id = self.particle_filter_frame
        #odom.header.frame_id = self.particle_filter_frame

        # Set position based on average particle pose
        odom.pose.pose.position.x = avg_pose[0]
        odom.pose.pose.position.y = avg_pose[1]
        odom.pose.pose.position.z = 0.

        # Set position based on average particle orientation
        yaw = atan2(s_thetas, c_thetas)
        quat = quaternion_from_euler(0, 0, yaw)
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        # Publish odometry message
        self.odom_pub.publish(odom)

        # get current odom --> left_cam T matrix
        # initialize transform
        tf = TransformStamped()

        # setup header
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "/map"

        # set child
        tf.child_frame_id = "/base_link"

        # set position
        tf.transform.translation.x = avg_pose[0]
        tf.transform.translation.y = avg_pose[1]
        tf.transform.translation.z = 0.

        # set orientation
        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]

        # broadcast transform
        self.tf_broadcaster.sendTransform(tf)

        # Initialize pose array to publish particles
        particle_msg = PoseArray()

        # Initialize particle poses
        poses = []

        # Iterate through particles
        for p in self.particles:

            # Initialize pose message
            pose_i = Pose()

            # Set particle position
            pose_i.position.x = p[0]
            pose_i.position.y = p[1]
            pose_i.position.z = 0.

            # Set particle orientation
            quat = quaternion_from_euler(0, 0, p[2])
            pose_i.orientation.x = quat[0]
            pose_i.orientation.y = quat[1]
            pose_i.orientation.z = quat[2]
            pose_i.orientation.w = quat[3]

            # Append particles to pose array
            poses.append(pose_i)
        
        # Set frame id to map
        particle_msg.header.frame_id = "/map"
        #particle_msg.header.frame_id = self.particle_filter_frame

        # Set particles poses
        particle_msg.poses = poses

        # Publish particles
        self.particles_pub.publish(particle_msg)

    # Whenever you get odometry data use the motion model to update the particle positions
    def odom_callback(self, msg):
        
        # Get the current time
        current_time = self.get_clock().now().nanoseconds

        # Calculte dt
        dt = (current_time - self.prev_time) * 1e-9

        # Save the current time as previous time for next update
        self.prev_time = current_time

        # Approximate dx, dy, and dtheta based on dt and velocities
        dx = msg.twist.twist.linear.x * dt
        dy = msg.twist.twist.linear.y * dt
        dtheta = msg.twist.twist.angular.z * dt

        # Update particle positions if particles have been initialized based on odom
        if self.particles is not None:
            self.particles = self.motion_model.evaluate(self.particles, [-dx, -dy, -dtheta])
            
            # Publish average pose and particle updates
            self.averager()

    # Whenever you get sensor data use the sensor model to compute the particle probabilities. 
    # Then resample the particles based on these probabilities
    def laser_callback(self, msg):

        # Increase count to determine next update
        self.laser_ct += 1

        # Update particle positions if particles have been initialized based on lidar
        if self.particles is not None and not self.laser_ct % self.laser_rate:

            # Get likelihood table
            #print("ranges len", len(msg.ranges))
            ranges_i = np.array(msg.ranges)[np.linspace(0,len(msg.ranges), self.sensor_model.num_beams_per_particle,endpoint=False,dtype=np.int32)]
            self.likelihood_table = self.sensor_model.evaluate(self.particles, ranges_i) #**self.pow_value
            #self.get_logger().info(f"Ranges  = {ranges_i}")
            #self.get_logger().info(f"Likelihood = {self.likelihood_table}")

            if self.likelihood_table is not None:
                self.likelihood_table /= np.sum(self.likelihood_table)

                # Get indicies from which to sample
                resample_inds = np.random.choice(a=self.num_particles, size=self.num_particles, p=self.likelihood_table)

                # Resample particles based on weights
                self.particles = self.particles[resample_inds, :]

                # Publish average pose and particle updates
                self.averager()

                # self.get_logger().info("LASER UPDATE")
            else:
                self.get_logger().info("Oh no")
    # Initialize pose based on 2D Pose estimate
    def pose_callback(self, msg):

        # Create odom object and set frame to map
        odom = Odometry()
        odom.header.frame_id = "/map"
        odom.child_frame_id = "/base_link_pf"

        pose = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        # Set position and orientation based on click
        odom.pose.pose.position = pose
        odom.pose.pose.orientation = orient

        # Publish pose
        self.odom_pub.publish(odom)

        # Generate random normal distribution around x, y of click
        std = 0.3
        x_samples = np.random.normal(pose.x, std, (self.num_particles - 1,1)) #+ msg.pose.pose.position.x
        y_samples = np.random.normal(pose.y, std, (self.num_particles - 1,1)) #+ msg.pose.pose.position.y

        # Get yaw value of click
        x = orient.x
        y = orient.y
        z = orient.z
        w = orient.w
        r, p, yaw = euler_from_quaternion([x, y, z, w])

        # Generate theta values pulled from a uniform distribution of +- 15°
        theta_samples = np.random.uniform(-pi/12, pi/12, (self.num_particles - 1,1)) + yaw
        #theta_samples = yaw * np.ones((self.num_particles, 1))

        # Add one particle at exactly the click pose
        x_samples = np.vstack((x_samples, pose.x))
        y_samples = np.vstack((y_samples, pose.y))
        theta_samples = np.vstack((theta_samples, yaw))

        # Set particles
        self.particles = np.hstack((x_samples, y_samples, theta_samples))

        self.get_logger().info("PARTICLES INITIALIZED")

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
