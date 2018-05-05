import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
  return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params, render=False):
    observation = env.reset()
    done = False
    t = 0

    while not done:
        if render:
            env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break

    return t

def play_multiple_episodes(num, env):
    best_t = float('-inf')
    best_params = None
    best_index = float('-inf')
    for i in range(num):
        params = np.random.rand(4)
        t = play_one_episode(env, params)
        if t > best_t:
            best_t = t
            best_params = params
            best_index = i
    print('Training finished')
    print('Best Params {} with t= {} and index= {}'.format(best_params, best_t, best_index))
    play_one_episode(env, best_params, render=True)


# [ 0.65256228  0.57807466  0.95957457  0.44585728]
env = gym.make('CartPole-v0')
env._max_episode_steps = 100000
play_multiple_episodes(1000, env)
env.close()