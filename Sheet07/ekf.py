import numpy as np
import matplotlib.pylab as plt

observations = np.load('data/observations.npy')


def get_observation(t):
    return observations[t]


class ExtendedKalmanFilter(object):
    def __init__(self, sigma_p, sigma_m):
   
        self.sigma_p = sigma_p
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None

        self.count = 0
        self.state_smoothed = []

    def init(self, init_state):
        # to do
        pass

    def track(self, xt):
        # to do
        pass

    def get_current_location(self):
        # to do
        pass

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track


def get_observations_up_to(n, datalist):
    x_1 = [x[0] for x in datalist[:n+1]]
    y_1 = [x[1] for x in datalist[:n+1]]

    return x_1, y_1 

def main():
    init_state = np.array([0, 0])

    sp = 0.01
    sigma_p = np.array([[sp, 0],
                        [0, sp],])

    sm = 0.001
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    tracker = ExtendedKalmanFilter(sigma_p, sigma_m)
    tracker.init(init_state)

    track = perform_tracking(tracker)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    num_steps = len(observations) 
    for n in range(num_steps):
        ax.clear()
        o_x, o_y = get_observations_up_to(n, observations) 
        t_x, t_y = get_observations_up_to(n, track) 

         
        ax.plot(o_x, o_y, 'g', label='observations')
        ax.plot(t_x, t_y, 'y', label='EKF')
        ax.legend()
        plt.pause(0.1)
    plt.pause(3)



if __name__ == "__main__":
    main()
