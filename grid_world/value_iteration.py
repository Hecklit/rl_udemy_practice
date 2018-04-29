import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from utils import print_policy, print_values

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
SMALL_ENOUGH = 10e-4
GAMMA = 0.9

if __name__ == '__main__':
    # random initalize V and pi
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