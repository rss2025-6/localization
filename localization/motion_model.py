import numpy as np

e=np.e
cos = np.cos
sin = np.sin

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.deterministic = False

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        new_particles = np.empty(particles.shape)
        del_pos = np.array([odometry[:2]]).T
        theta_w_r = odometry[2]

        # T_odom = np.array([[cos(theta_w_r), -sin(theta_w_r)], [sin(theta_w_r), cos(theta_w_r)]])
        # T_odom = np.hstack((T_odom, del_pos))
        # T_odom = np.vstack((T_odom, np.array([[0,0,1]])))

        for i in range(len(particles)):
            if self.deterministic:
                T_odom = np.array([[cos(theta_w_r), -sin(theta_w_r)], [sin(theta_w_r), cos(theta_w_r)]])
                T_odom = np.hstack((T_odom, del_pos))
                T_odom = np.vstack((T_odom, np.array([[0,0,1]])))
            else:
                rng = np.random.default_rng()
                variance = .005 # Change this to increase or decrease the noise distribution
                new_theta_w_r = theta_w_r + rng.normal(0, variance, None)
                T_odom = np.array([[cos(new_theta_w_r), -sin(new_theta_w_r)], [sin(new_theta_w_r), cos(new_theta_w_r)]])
                T_odom = np.hstack((T_odom, del_pos))
                T_odom = np.vstack((T_odom, np.array([[0,0,1]])))
                T_odom[0,2] += rng.normal(0, variance, None)
                T_odom[1,2] += rng.normal(0, variance, None)

                #np.array([[rng.normal(0, variance, None)], [rng.normal(0, variance, None)]])

            pose = particles[i]
            x_k1 = np.array([pose[:2]]).T
            theta_t1 = pose[-1]
            R_w_r = np.array([[cos(theta_t1), -sin(theta_t1)], [sin(theta_t1), cos(theta_t1)]])

            T_particle = np.hstack((R_w_r, x_k1))

            T_particle = np.vstack((T_particle, np.array([[0,0,1]])))

            T_k = T_particle @ T_odom
            # x_update = np.array([*T_k[:2,2], np.arctan2(T_k[1,0], T_k[0,0])])

            # new_particles[i]=x_update
            new_particles[i, 0] = T_k[0,2]
            new_particles[i, 1] = T_k[1,2]
            new_particles[i, 2] = np.arctan2(T_k[1,0], T_k[0,0])

        return new_particles
        ####################################







