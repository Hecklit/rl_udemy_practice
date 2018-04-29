import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from utils import print_policy, print_values

SMALL_ENOUGH = 10e-4


V = {}
grid = standard_grid()
S = grid.all_states()
for state in S:
    V[state] = 0
gamma = 1.0
while True:
    delta = 0
    for s in S:
        old_v = V[s]
        
        if s in grid.actions:
            new_v = 0
            p_a = 1.0 / len(grid.actions[s])
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                new_v += p_a * (r + gamma * V[grid.current_state()])
            V[s] = new_v
            delta = max(delta, np.abs(V[s] - old_v))
    if delta < SMALL_ENOUGH: break
print('Values for uniform random actions:')
print_values(V, grid)
print('\n\n')



policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U'
}
print_policy(policy, grid)

V = {}
grid = standard_grid()
S = grid.all_states()
for state in S:
    V[state] = 0
gamma = 0.9
while True:
    delta = 0
    for s in S:
        old_v = V[s]
        
        if s in grid.actions:
            new_v = 0
            p_a = 1.0
            a = policy[s]
            grid.set_state(s)
            r = grid.move(a)
            V[s]= p_a * (r + gamma * V[grid.current_state()])
            delta = max(delta, np.abs(V[s] - old_v))
    if delta < SMALL_ENOUGH: break
print('Values for fixed policy actions:')
print_values(V, grid)
print('\n\n')