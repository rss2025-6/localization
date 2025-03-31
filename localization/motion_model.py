import numpy as np

cos = np.cos
sin = np.sin

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

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
        # TODO

        # x_k-1 = particles[-1]
        # del_x = odom

        for k1 in particles:
            del_pos = np.array([odometry[:2]]).T
            theta_w_r = odometry[2]

            # x_k1 = particles[-1]
            x_k1 = particles[k1]
            theta_t1 = x_k1[2, 0]
            R_w_r = np.array([[cos(theta_t1), -sin(theta_t1)], [sin(theta_t1), cos(theta_t1)]])
            pos_new = R_w_r @ del_pos + x_k1[:2]
            x_update = np.vstack((pos_new, x_k1[2] + odometry[2]))

            particles[k1] = x_update #.T[0]

        ####################################







