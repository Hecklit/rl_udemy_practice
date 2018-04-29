import numpy as np

class Agent():
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []
        self.V = None
        self.sym = None

    def setV(self, V):
        self.V = V
    
    def set_symbol(self, sym):
        self.sym = sym

    def set_verbose(self, v):
        self.verbose = v
    
    def reset_history(self):
        self.state_history = []


    def take_action(self, env):
        best_state = None
        r = np.random.rand()
        if r <= self.eps:
            if self.verbose:
                print('Taking a random action')

            possible_moves = []
            for i in range(3):
                for j in range(3):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}
            next_move = None
            best_value = -1
            for i in range(3):
                for j in range(3):
                    if env.is_empty(i, j):
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env.board[i, j] = 0
                        pos2value[(i,j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)
            if self.verbose:
                print('Taking a greedy action')
                for i in range(3):
                    print('-'*16)
                    row = ' '
                    for j in range(3):
                        value = env.board[i, j]
                        x1 = 'x   |' if value == env.x else ('o   |' if value == env.o else '{:00.2f}|'.format(pos2value[(i, j)]))
                        row += '{}'.format(x1)
                    row += ' '
                    print(row)
                print('-'*16)
        env.board[next_move[0], next_move[1]] = self.sym
                    
        
    def update_state_history(self, s):
        self.state_history.append(s)

    def update(self, env):
        # print('Updating {}'.format(self.sym))
        reward = env.reward(self.sym)
        target = reward
        # print('Reward: {}'.format(reward))
        for prev in reversed(self.state_history):
            # print('State ({})'.format(prev))
            # print('value = self.V[prev] + self.alpha*(target - self.V[prev])')
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            # print('{} = {} + {}*({} - {})'.format(value, self.V[prev], self.alpha, target, self.V[prev]))
            self.V[prev] = value
            target = value
        self.reset_history()