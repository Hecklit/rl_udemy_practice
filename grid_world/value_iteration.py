import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
SMALL_ENOUGH = 10e-4
GAMMA = 0.9

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
            row += '{}:  {}   |'.format((i, j), a)
        print(row)

def wind(a):
    urne = ALL_POSSIBLE_ACTIONS + [a, a]
    return np.random.choice(urne);


if __name__ == '__main__':
    # random intalize V and pi
    V = {}
    grid = negative_grid(-1.901)
    print('Rewards:')
    print_values(grid.rewards, grid)

    S = grid.all_states()
    for state in S:
        V[state] = 0

    pi = {}
    for state in grid.actions.keys():
            pi[state] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print('Inital Policy')
    print_policy(pi, grid)
    print('Inital Vs')
    print_values(V, grid)

    iteration = 0
    while True:
        iteration += 1
        # iterative policy evaluation
        delta = 0
        for s in S:
            old_v = V[s]
            new_v = 0
            best_v = float('-inf')
            if s in pi:
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v = r + GAMMA * V[grid.current_state()]
                    if new_v > best_v:
                        best_v = new_v
                V[s] = best_v
                delta = max(delta, np.abs(V[s] - old_v))
        print('*'*60)
        print('Iteration {}'.format(iteration))
        print_values(V, grid)
        print_policy(pi, grid)
        if delta < SMALL_ENOUGH: break
    
    for s in grid.actions.keys():
        best_v = float('-inf')
        best_a = None
        if s in pi:
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                new_v = r + GAMMA * V[grid.current_state()]
                if new_v > best_v:
                    best_v = new_v
                    best_a = a
            pi[s] = best_a

    print_values(V, grid)
    print_policy(pi, grid)