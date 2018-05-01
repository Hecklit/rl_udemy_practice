
def print_values(V, g):
    for i in range(g.width):
        print('-'*28)
        row = ''
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                row += ' {:0.2f} |'.format(v)
            else:
                row += '{:0.2f} |'.format(v)
        print(row)
        
def print_policy(P, g):
    for i in range(g.width):
        print('-'*28)
        row = ''
        for j in range(g.height):
            a = P.get((i, j), ' ')
            row += '  {}   |'.format(a)
        print(row)

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
import numpy as np

def random_action(a, eps=0.1):
    p = np.random.rand()
    if p < (1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS);