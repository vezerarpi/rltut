# coding: utf-8

# In[ ]:


import os
import random

import gym
from gym import wrappers
import chainer as C
import numpy as np


# In[ ]:


exp_init_size = 500
n_episodes = 8000
# number of episodes after which to print q function evaluation
eval_period = 20
rng = random.Random()
rng.seed(42)


# In[ ]:


class Model(C.Chain):
    def __init__(self):
        super().__init__()
        state_size = 4
        action_size = 2
        hidden_size = 16
        # Chain.init_scope is necessary for gradient book-keeping to be set up
        # for all the links defined below, otherwise errors are not
        # propagated back through the graph
        with self.init_scope():
            self.l1 = C.links.Linear(state_size, hidden_size)
            self.l2 = C.links.Linear(hidden_size, action_size)

    def __call__(self, state):
        h = C.functions.relu(self.l1(state))
        return self.l2(h)


# In[ ]:


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
        self._model = Model()
        if self._update_freq > 1:
            self._target_model = Model()
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
        return np.asscalar(loss.data), loss#XXX loss returned from printing link

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


agent = Agent()
print([agent.act(np.ones(4, dtype=np.float32)) for i in range(50)])
agent.reward(1.0, np.ones(4, dtype=np.float32))

# In[ ]:

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
