import numpy as np

class Bandit:
    def __init__(self, id, prob_of_success, mean=0):
        self.id = id
        self.prob_of_success = prob_of_success
        self.mean = mean
        self.num_pulls = 0.001

    def pull(self):
        res = np.random.rand() <= self.prob_of_success
        self.num_pulls += 1
        self.mean = self.iterative_mean(res)
        return res

    def iterative_mean(self, new_element):
        num = int(new_element)
        assert num in [0, 1], 'New Element is neither 0 nor 1'
        new_mean = (1-(1/self.num_pulls)) * self.mean + (1/self.num_pulls) * num
        return new_mean

    def __repr__(self):
        return 'I am Bandit {} with prob_s {} num pulls {} and mean {}'.format(self.id, self.prob_of_success, self.num_pulls, self.mean)