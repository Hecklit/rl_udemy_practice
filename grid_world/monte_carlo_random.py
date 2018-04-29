import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from utils import print_policy, print_values

SMALL_ENOUGH = 10e-4
GAMMA = 0.9

def wind(a):
    return np.random.choice(ALL_POSSIBLE_ACTIONS + [a, a])

def play_episode(grid, pi):
    
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = pi[s]
        a = wind(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append([s, G])
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns

def first_visit_monte_carlo_prediction(pi, N):
    V = {}
    all_returns = {} # default = []
    for i in range(N):
        visited_states = set()
        states_and_returns = play_episode(standard_grid(), pi)
        for s, g in states_and_returns:
            if s not in visited_states:
                visited_states.add(s)
                if s not in all_returns:
                    all_returns[s] = []
                all_returns[s].append(g)
                V[s] = np.mean(all_returns[s])
    return V

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
if __name__ == '__main__':
    pi = {
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 2): 'U',
        (2, 1): 'L',
        (2, 0): 'U',
        (1, 0): 'U',
        (2, 3): 'L',
    }
    grid = standard_grid()
    print_policy(pi, grid)
    V = first_visit_monte_carlo_prediction(pi, 100)
    print_values(V, grid)