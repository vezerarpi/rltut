'''
# Deep Reinforcement Learning Tutorial

See the `notes.ipynb` notebook for a brief overview of the theory behind Q-learning.
'''


'''
## Preliminaries

Dependencies and setup
'''
import random
import string

import chainer as C
import gym
import numpy as np

from .rltut import model


# TODO replace the output path with your name, make sure it's unique
output_path = 'out/REPLACE-ME-' + ''.join(random.choice(string.ascii_lowercase) for i in range(6))
rng = random.Random()
rng.seed(42)


'''
## OpenAI Gym

OpenAI Gym wraps a range of simple RL environments in a fairly easy to use module. We will start with the most basic one, CartPole, which has a simple enough state space to be easily visualised but still has some interesting dynamics for an agent to learn.

The `gym.make()` function creates environment by name, in this case `'CartPole-v0'`. Once an environment has been created we can query it for its observation and action spaces and instruct actions to be taken in it to get a reward and the next state. The environment has to be reset to start get an initial state and start what is called an episode. An episode is a series of steps in the environment, ended when a terminal state is reached or we next call reset (it often makes sense to have a max number of steps even if a terminal state hasn't been reached yet).

* `state = env.make(name)` creates an environment
* `state = env.reset()` resets the environment, starts a new episode and returns an initial state observation
* `next_state, reward, done = env.step(action)` performs the action in the environment and returns the next state, the reward for the action and whether current episode has ended
* The action and observation spaces have a `sample()` method to sample random states and examples

'''
env = gym.make('CartPole-v0')
print('observation space', env.observation_space)
print('action space', env.action_space, 'action_space.n', env.action_space.n)
state = env.reset()
print('state', state)

'''
## CartPole

The task in CartPole is to keep upright a pole that is balanced on a cart by moving the cart either left or right. The episode ends if the pole swings past 12 degrees from upright or if the cart moves out of bounds.

The environment rewards a value of 1 for each time step taken, including on the final timestep that ends the episode. The goal is therefore to keep the pole within bounds for a long as possible, up to 200 steps.

We generally don't need to know the exact details of the state or the action spaces and just need to know the sizes so that an RL algorithm can know what input it takes and how many actions to select from. The state for cartpole is a quadruple: `[cart_position, cart_velocity, pole_theta, pole_angular_velocity]`. The pole angle and velocity is in radians, vertical is 0.0, left of centre is negative. For an Atari game we would just have a 3D matrix containing the RGB values of the current state of the screen, H x W x (R, G, B), but more on that later.
'''
for ep in range(5):
    state = env.reset()
    done = False
    reward = 0.0
    actions = list(range(env.action_space.n))
    steps = 0
    while not done:
        state, r, done = env.step(random.choice(actions))
        reward += r
        steps += 1
    print(ep, 'steps', steps, 'reward', reward, 'final state', state)
    print(['-'] * 5)

'''
## Video output

Gym can output videos of episodes, which for CartPole can be played in real time. This is useful for seeing how well an algorithm has learnt to play the game. This is easily done by wrapping the environment in a `Monitor`.
* `directory` - output directory path, make sure you've personalised it above
* `force` - flag to force creation of new, or overwriting of existsing output directires
* `video_callable` - a function that takes the episode number and returns whther to record a video, defaults to every cubic number or every 1K episodes after the first 1K
'''
env = gym.wrappers.Monitor(env, directory=output_path, force=True,
                           video_callable=lambda ep: ep % 10 == 0)
# TODO try the previous random actions here again but now with the monitor wrapping the env

'''
## Agent
'''


class Agent:
    def __init__(self):
        '''
        Create an agent containing
         * a model
         * an optimiser
        '''

    def act(self, state):
        '''
        Returns an action (as an int) for the current state. Keeps track of the state and the action taken for use in the next call to reward(). The action is chosen e-greedily (a random action is taken with probability e).

         - state a np.array representing the current observed state
         - return the selected action
        '''

    def reward(self, reward, next_state):
        '''
        Takes the reward for the alt action and the resulting next_state, calculates the Q-learning loss and performs a parameter update on the model.

         - reward a float, the reward for the latest act()
         - next_state a np.array containing the observed next state resulting
         from the latest act()
         - return None
        '''


'''
## Training loop

'''
n_episodes = 1000
# number of episodes after which to print q function evaluation
eval_period = 20

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, directory=output_path, force=True)
env.seed(0)
agent = Agent()

for ep in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        state, reward, done, _ = env.step(agent.act(np.array(state, dtype=np.float32)))
        state = np.array(state, dtype=np.float32)
        agent.reward(reward, None if done else state)
    if ep % eval_period == 0:
        ep_lengths = env.get_episode_lengths()[-eval_period:]
        print('-' * 11)
        if ep:
            print('episodes', ep - eval_period, '-', ep, 'steps', ' '.join(map(str, ep_lengths)))
