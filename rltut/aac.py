import argparse
import os
import random

import gym
from gym import wrappers
import chainer as C
import numpy as np


env = gym.make('CartPole-v0')

agent = None
n_episodes = 1000
max_steps = 200
env.seed(0)
rng = random.Random()
rng.seed(42)

for ep in range(n_episodes):
    state = env.reset()
    for step in range(max_steps):
