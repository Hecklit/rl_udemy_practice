import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

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
            row += '  {}   |'.format(a)
        print(row)


def iterative_policy_evaluation(pi, V, grid):
    S = grid.all_states()

    while True:
        delta = 0
        for s in S:
            if s in pi:
                old_v = V[s]
                a = pi[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s]= r + GAMMA * V[grid.current_state()]
                delta = max(delta, np.abs(V[s] - old_v))
        if delta < SMALL_ENOUGH: break
    return V

# random intalize V and pi
V = {}
grid = negative_grid(-1)
print_values(grid.rewards, grid)

S = grid.all_states()
for state in S:
    if grid.is_terminal(state):
        V[state] = 0
    else:
        V[state] = np.random.rand()

pi = {}
for state in S:
    if not grid.is_terminal(state):
        pi[state] = np.random.choice(grid.actions[state])

print_policy(pi, grid)
print_values(V, grid)

iteration = 0
while True:
    print(iteration)
    iteration += 1
    # iterative policy evaluation
    V = iterative_policy_evaluation(pi, V, grid)

    # 3 step
    policy_changed = False
    for s in S:
        if s in pi:
            old_a = pi[s]
            best_value = float('-inf')
            best_action = None
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                v = r + GAMMA * V[grid.current_state()]
                if v > best_value:
                    best_action = a
                    best_value = v
            if old_a != best_action:
                #print('in State {} best actions was {} now its {} with value {}'.format(s, old_a, best_action, best_value))
                pi[s] = best_action
                policy_changed = True
    if not policy_changed:
        break

print_values(V, grid)
print_policy(pi, grid)