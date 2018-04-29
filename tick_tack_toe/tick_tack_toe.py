import numpy as np
class TickTackToe:

    def __init__(self):
        self.board = np.zeros((3,3))
        self.x = 1
        self.o = -1
        self.ended = False
        self.winner = None
        self.num_states = 3**9

    def is_empty(self, i, j):
        return self.board[i, j] == 0

    def reward(self, symbol):
        if not self.game_over():
            return 0
        return 1 if self.winner == symbol else 0

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended
        # horizontal
        for x in range(3):
            for player in (self.x, self.o):
                if self.board[x].sum() == player * 3:
                    self.winner = player
                    self.ended = True
                    return True
        # vetical
        for x in range(3):
            for player in (self.x, self.o):
                if self.board[:,x].sum() == player * 3:
                    self.winner = player
                    self.ended = True
                    return True
        # diagonals
        for player in (self.x, self.o):
            if self.board.trace() == player * 3:
                self.winner = player
                self.ended = True
                return True
            if np.fliplr(self.board).trace() == player * 3:
                self.winner = player
                self.ended = True
                return True
        # draw
        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True

        self.winner = None
        return False

    def get_state(self):
        k = 0
        h = 0
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    v = 0
                if self.board[i, j] == self.x:
                    v = 1
                if self.board[i, j] == self.o:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

    def draw_board(self):
        for i in range(3):
            print('-'*16)
            row = ' '
            for j in range(3):
                value = self.board[i, j]
                x1 = 'x   |' if value == self.x else ('o   |' if value == self.o else '    |')
                row += '{}'.format(x1)
            row += ' '
            print(row)
        print('-'*16)