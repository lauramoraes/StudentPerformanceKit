import numpy as np
from datetime import datetime


class SimulateStudent(object):
    def __init__(self, p_L0, p_T, p_S, p_G, timestamp_prior=None):
        self.p_L0 = p_L0
        self.p_T = p_T
        self.p_S = p_S
        self.p_G = p_G
        self.timestamp_prior = timestamp_prior

    def simulate(self, nSteps):
        """ given an HMM = (A, B, pi), simulate state and observation sequences """
        observations = np.zeros(nSteps, dtype=np.int) # array of zeros
        # Probability of being in the learned state at the beginning
        p_L = self.p_L0
        #print(p_L)
        for t in range(0, nSteps): # loop through t
            # Probability of getting next question correctly
            p_corr = (1 - p_L) * p_G + p_L * (1 - p_S)
            #print("p corr %f" % p_corr)
            # Draw next outcome
            observations[t] = np.random.binomial(1, p_corr)
            # Update learning state probability
            if observations[t]:
                learning_evidence = p_L * (1 - self.p_S) / (p_L * (1 - self.p_S) + (1 - p_L) * self.p_G)
            else:
                learning_evidence = p_L * self.p_S / (p_L * self.p_S + (1 - p_L) * (1 - self.p_G))
            p_L = learning_evidence + (1 - learning_evidence) * self.p_T
            #print(p_L)
        return observations

    def simulate_timestamp(self, state, time_steps, last_time=datetime(2019,12,17)):
        ts_prior = self.timestamp_prior[state]
        idx =  self.random_MN_draw(1, ts_prior)
        random_timedelta = time_steps[idx]
        random_ts = last_time + np.random.random() * random_timedelta
        return random_ts
