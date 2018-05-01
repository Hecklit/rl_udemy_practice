import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from utils import print_policy, print_values, random_action
# np.random.seed(42)

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALPHA = 2.0
ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']

def greedy_from(Qs):
    best_action = None
    best_value = float('-inf')
    for action in Qs:
        value = Qs[action]
        if value > best_value:
            best_action = action
            best_value = value 
    return best_action, best_value


if __name__ == '__main__':
    grid = negative_grid(-0.5)
    print('rewards')
    print_values(grid.rewards, grid)
    S = grid.all_states()
    Q = {}
    num_seen_sa = {}
    for s in S:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            num_seen_sa[(s, a)] = 0
            if grid.is_terminal(s):
                Q[s][a] = 0
            else:
                Q[s][a] = np.random.rand()
    N = 10000
    deltas = []

    for t in range(1, N):
        if t%1000 == 0:
            print(t)
        s = (2, 0)
        grid.set_state(s)
        a, _ = greedy_from(Q[s])
        a = random_action(a, eps=0.1)
        while not grid.game_over():
            r = grid.move(a)
            s_prime = grid.current_state()
            a_prime, _ = greedy_from(Q[s_prime])
            a_prime = random_action(a_prime, eps=(1.0/t))
            q_sa = Q[s][a]
            num_seen_sa[(s, a)] += 1
            # print('Learning Rate: ', ALPHA/num_seen_sa[(s, a)])
            Q[s][a] = q_sa + (ALPHA/num_seen_sa[(s, a)]) * (r + GAMMA*Q[s_prime][a_prime] - q_sa)
            delta = np.abs(Q[s][a] - q_sa)
            deltas.append(delta)
            s = s_prime
            a = a_prime

    pi = {}
    for s in S:
        if not grid.is_terminal(s):
            pi[s], _ = greedy_from(Q[s])
    print('Results:')
    print_policy(pi, grid)
    print(len(deltas))
    plt.plot(deltas)
    plt.show()
