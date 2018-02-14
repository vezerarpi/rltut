'''
# Deep Reinforcement Learning Tutorial

## Introduction

### Types of ML
#### Supervised and unsupervised learning
The loss or objective function that we want to optimise for is known in both of these settings. For supervised learning we have a dataset of *(input, output)* pairs, our models make predictions for the output values based on the inputs and then a loss function scores how good a prediction this was compared to the *correct* output.

Similarly for unsupervised learning, while there isn't a predefined output label for each input, there is a known objective function that scores some derived property of the data. E.g. word embeddings are trained to minimise the distance between the vectors representing similar words, or for language modelling there is a loss function (softmax cross-entropy) scoring how well the model can predict each word in the dataset from the words that came before it.

#### Reinforcement learning
Unlike the previous two scenarios, there isn't a known objective function that we can directly optimise for. RL problems are usually framed as an agent interacting with an environment. The agent observes states, takes actions and recieves rewards from the environment but the inner workings or the true dynamics driving the environment cannot be directly observed.

The agent observes the current state of the environment and chooses and action, in response to which the environment provides a reward and the next state that result from the action. RL algorithms try to maximise the rewards for the agent by learning better and better choices of actions.

Both the agent and environment can be stochastic.
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
print('action space', env.action_space, 'action_space.n', action_space.n)
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
#TODO try the previous random actions here again but now with the monitor wrapping the env

#XXX
#XXX continue writing here
#XXX

class Agent:
    def __init__(self):
        # number of steps after which to update parameters of the target model
        self._update_freq = 1
        self._lr = 0.01
        self._epsilon = 1.0
        self._gamma = 0.95
        # experience buffer
        # {'state', 'action', 'reward', 'next_state'}
        self._exps = []
        self._last_act = 0
        self._state = None
        self._model = model.CartPoleModel()
        if self._update_freq > 1:
            self._target_model = model.CartPoleModel()
            self._target_model.copyparams(self._model)
        else:
            self._target_model = self._model
        self._update_count = 0
        self._optim = C.optimizers.SGD(lr=self._lr)
        # self._optim = C.optimizers.RMSprop(lr=self._lr)
        self._optim.setup(self._model)

    def act(self, state):
        if rng.random() < self._epsilon:
            self._last_act = rng.choice([0, 1])
        else:
            x = state.reshape((1, -1))
            self._last_act = np.argmax(self._model(C.Variable(x)).data)
        self._state = state
        return self._last_act

    def _target(self, experience):
        # y = reward + gamma * Qmax(action, next_state)
        y = np.float32(experience['reward'])
        if experience['next_state'] is not None:
            # next state x
            x = experience['next_state'].reshape((1, -1)).astype(np.float32)
            y += self._gamma * np.max(self._target_model(C.Variable(x)).data)
        return y

    def _make_exp(self, state, action, reward, next_state):
        return dict(state=state, action=action, reward=reward, next_state=next_state)

    def store(self, state, action, reward, next_state):
        self._exps.append(self._make_exp(state, action, reward, next_state))

    def reward(self, reward, next_state):
        self._epsilon -= 1e-3
        self._exps.append(self._make_exp(self._state, self._last_act, reward, next_state))
        batch_size = 64
        # sample a batch
        sample = rng.sample(self._exps, k=min(batch_size, len(self._exps)))
        # eval batch
        states = np.stack([s['state'] for s in sample])
        actions = np.stack([s['action'] for s in sample]).astype(np.int32)
        y = C.Variable(np.stack([self._target(s) for s in sample]).astype(np.float32))
        # calc loss
        self._model.cleargrads()
        qs = self._model(C.Variable(states))
        q = qs[np.arange(len(sample)), actions]
        loss = C.functions.mean_squared_error(y, q)
        loss.backward()
        self._optim.update()
        self._update_count += 1
        if self._update_freq > 1 and self._update_count % self._update_freq == 0:
            self._target_model.copyparams(self._model)
        return np.asscalar(loss.data)

    def eval_q(self, states):
        qs = self._model(C.Variable(states)).data
        self._model.cleargrads()
        return qs


def eval_states(agent):
    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L31 noqa
    theta_limit = 10 * 2 * np.pi / 360
    n = 7
    thetas = np.flip(np.linspace(-theta_limit, theta_limit, n), axis=0)
    states = np.array([[0.0, 0.0, theta, 0.0]
                       for theta in thetas],
                      dtype=np.float32)
    print('Eval Theta', ''.join(['[{:^12.1f}]'.format(x)
                                 for x in states[:, 2] * 360 / 2 / np.pi]))
    qs = agent.eval_q(states)
    a = ['_R' if x else 'L_' for x in np.argmax(qs, axis=1)]
    print('Eval L - R', ''.join(['[{:^12.1f}]'.format(l - r)
                                 for l, r in qs]))
    print('Eval Q    ', ''.join(['[{:5.2f}{}{:5.2f}]'.format(l, y, r)
                                 for (l, r), y in zip(qs, a)]))


def print_params(chain):
    for path, param in chain.namedparams():
        print(path, param)


exp_init_size = 500
n_episodes = 8000
# number of episodes after which to print q function evaluation
eval_period = 20

agent = Agent()
print([agent.act(np.ones(4, dtype=np.float32)) for i in range(50)])
agent.reward(1.0, np.ones(4, dtype=np.float32))

env = gym.make('CartPole-v0')
env.seed(0)
agent = Agent()

state = None
for i in range(exp_init_size):
    if state is None:
        state = env.reset()
        state = np.array(state, dtype=np.float32)
    action = rng.choice([0, 1])
    next_state, reward, done, _ = env.step(action)
    next_state = None if done else np.array(next_state, dtype=np.float32)
    agent.store(state, action, reward, next_state)
    state = next_state

env = gym.wrappers.Monitor(env, directory='out', force=True)

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
        eval_states(agent)
