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
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    states_actions_rewards = [(s, a, 0)]
    seen_states = set()
    while True:
        r = grid.move(a)
        s = grid.current_state()

        if s in seen_states:
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = pi[s]
            # a = wind(a)
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)

    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append([s, a, G])
        G = r + GAMMA * G
    states_actions_returns.reverse()
    # print(states_actions_returns)
    # print('*'*60)
    return states_actions_returns

def max_dict(d):
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
if __name__ == '__main__':

    grid = negative_grid()
    print('rewards:')
    print_values(grid.rewards, grid)

    pi = {}
    for s in grid.actions.keys():
        pi[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    Q = {}
    returns = {}
    S = grid.all_states()
    #initalize Q and returns
    for s in S:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []

    

    print_policy(pi, grid)
    deltas = []
    for i in range(10000):
        if i% 1000 == 0:
            print(i)

        delta = 0
        # Policy Evaluation Step
        states_actions_returns = play_episode(grid, pi)
        seen_state_actions_pairs = set()
        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_actions_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                delta = max(delta, np.abs(old_q - Q[s][a]))
                seen_state_actions_pairs.add(sa)
        deltas.append(delta)

        # Policy Improvement Step
        for s in pi.keys():
            pi[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print('Done')
    print_policy(pi, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
    print_values(V, grid)