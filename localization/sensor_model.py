import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)

pi = np.pi
e = np.e
sqrt = np.sqrt

class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = self.table_width - 1
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
        

    def p_hit(self, z, d, eta = 1):
        if 0 <= z <= self.z_max:
            return eta * e**(-(z - d)**2 / (2 * self.sigma_hit**2)) / sqrt(2 * pi * self.sigma_hit**2)

        return 0

    def p_short(self, z, d):
        if 0 <= z <= d and d != 0:
            return 2 * (1 - z/d)/d
        
        return 0

    def p_max(self, z, d, eps = 1):
        if z == self.z_max:
            return 1
        
        return 0

    def p_rand(self, z, d):
        if 0 <= z <= self.z_max:
            return 1/self.z_max
        
        return 0

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        # rows = i = z values
        # cols = j = d values
        p_hits = np.zeros((len(self.sensor_model_table), len(self.sensor_model_table[0])))
        for i in range(len(self.sensor_model_table)):
            
            for j in range(len(self.sensor_model_table[0])):
                p_update = self.alpha_short * self.p_short(i, j) + self.alpha_max * self.p_max(i, j) + self.alpha_rand * self.p_rand(i, j)
                self.sensor_model_table[i][j] = p_update
                p_hits[i,j] = self.p_hit(i,j)

        norms = np.sum(p_hits, axis=0, keepdims = True)
        p_hits /= norms
        p_hits *= self.alpha_hit

        self.sensor_model_table += p_hits

        
        self.sensor_model_table/=np.sum(self.sensor_model_table, axis=0,keepdims=1)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # simulated scans for each particle
        scans = self.scan_sim.scan(particles)

        scale = self.resolution*self.lidar_scale_to_map_scale

        scans_px = scans/scale # do i need to do more to this since its Nxm
        obs_px = observation/scale

        # TODO CHECK THIS
        scans_px = np.clip(scans_px, 0, self.z_max)
        obs_px = np.clip(obs_px, 0, self.z_max)

        likelihood_table = np.ones((len(particles)))

        for i in range(len(scans)):
            for j in range(len(scans[0])):
                range_i = scans[i, j]
                likelihood_table[i] *= self.sensor_model_table[int(range_i), int(obs_px[j])]

        ####################################

        return likelihood_table
        # sensor_model_table[d_values = scans, z_values = observations]
        # return self.sensor_model_table[scans_px, obs_px]

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
