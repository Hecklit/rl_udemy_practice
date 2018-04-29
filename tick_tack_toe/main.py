import numpy as np
from agent import Agent
from human import Human
from tick_tack_toe import TickTackToe

def initialV_x(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def initialV_o(env, state_winner_triples):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i,j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j + 1)
    return results

def play_game(p1, p2, env, draw=False):
    current_player = None

    while not env.game_over():
        current_player = p2 if current_player == p1 else p1


        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        current_player.take_action(env)

        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
    if draw:
        env.draw_board()

    p1.update(env)
    p2.update(env)

if __name__ == '__main__':
    p1 = Agent()
    p2 = Agent()

    env = TickTackToe()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)

    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, TickTackToe())

    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, TickTackToe(), draw=2)

        answer = input('Play again? [Y/n]: ')
        if answer and answer.lower()[0] == 'n':
            break
