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

        self.deterministic = True

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
        if self.deterministic:
            noise = np.array([[0],[0]])
        else:
            rng = np.random.default_rng()
            variance = .1 # Change this to increase or decrease the noise distribution
            noise = np.array([[rng.normal(0, variance, None)], [rng.normal(0, variance, None)]])
    
        new_particles = np.empty(particles.shape)
        del_pos = np.array([odometry[:2]]).T
        theta_w_r = odometry[2]
        for i in range(len(particles)):
            pose = particles[i]
            x_k1 = np.array([pose[:2]]).T
            theta_t1 = pose[-1]
            R_w_r = np.array([[cos(theta_t1), -sin(theta_t1)], [sin(theta_t1), cos(theta_t1)]])
            x_k = R_w_r @ del_pos + x_k1 + noise
            x_update = np.vstack((x_k, theta_t1 +theta_w_r)).T
            new_particles[i]=x_update

        return new_particles
        ####################################







