import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from utils import print_policy, print_values

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
SMALL_ENOUGH = 10e-4
GAMMA = 0.9


def iterative_policy_evaluation(pi, V, grid):
    S = grid.all_states()

    while True:
        delta = 0
        for s in S:
            old_v = V[s]

            new_v = 0
            if s in pi:
                for a in ALL_POSSIBLE_ACTIONS:
                    if a == pi[s]:
                        p = 0.5
                    else:
                        p = 0.5/3
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p*(r + GAMMA * V[grid.current_state()])
                V[s] = new_v
                delta = max(delta, np.abs(V[s] - old_v))
        if delta < SMALL_ENOUGH: break
    return V


if __name__ == '__main__':
    # random intalize V and pi
    V = {}
    grid = negative_grid(-1.0)
    print('Rewards:')
    print_values(grid.rewards, grid)

    S = grid.all_states()
    for state in S:
        if grid.is_terminal(state):
            V[state] = 0
        else:
            V[state] = np.random.rand()

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
        V = iterative_policy_evaluation(pi, V, grid)
        print('*'*60)
        print('Iteration {}'.format(iteration))
        print_values(V, grid)
        
        # policy improvement step
        policy_changed = False
        for s in S:
            if s in pi:
                old_a = pi[s]
                best_value = float('-inf')
                best_action = None
                for a in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for a2 in ALL_POSSIBLE_ACTIONS:
                        if a == a2:
                            p = 0.5
                        else:
                            p= 0.5/3
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p*(r + GAMMA * V[grid.current_state()])
                    if v > best_value:
                        best_action = a
                        best_value = v
                pi[s] = best_action
                if old_a != best_action:
                    #print('in State {} best actions was {} now its {} with value {}'.format(s, old_a, best_action, best_value))
                    policy_changed = True
        if not policy_changed:
            break

    print_values(V, grid)
    print_policy(pi, grid)