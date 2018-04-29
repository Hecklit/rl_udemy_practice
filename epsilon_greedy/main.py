from bandit import Bandit
import numpy as np
import copy
from pprint import pprint
import matplotlib.pyplot as plt 
import math


num_bandits = 5
bandits = []

for i in range(num_bandits):
    bandits.append(Bandit(i, np.random.rand(), mean=0))

#epsilon = 0.01


def take_turn(epsilon, i, my_bandits):
    choosen_bandit = None
    if np.random.rand() <= epsilon:
        choosen_bandit = np.random.choice(my_bandits)
    else:
        choosen_bandit = max(my_bandits, key=lambda x: x.mean + math.sqrt(2*math.log(i+1)/x.num_pulls))
    return choosen_bandit.pull()
    #print('Turn {}: Choosen bandit was Bandit {}'.format(i, choosen_bandit.id))

epochs = 1000000
for eps in [0]:
    my_bandits = copy.deepcopy(bandits)
    start_bandits = copy.deepcopy(bandits)
    start_bandits.sort(key=lambda x: x.prob_of_success, reverse=True)
    reward = 0
    rewards = []
    for i in range(epochs):
        reward += take_turn(eps, i, my_bandits)
        rewards.append(reward)
    bandits.sort(key=lambda x: x.mean, reverse=True)
    print('Start')
    #pprint(start_bandits)
    print('End Bandits')
    pprint(my_bandits)
    print('Epsilon: {}'.format(eps))
    print('Epochs: {}'.format(epochs))
    print('Reward:', reward)