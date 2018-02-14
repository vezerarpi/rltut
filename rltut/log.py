import os
import re
import time
import numpy as np
import shutil
import gym
import json
import pandas as pd
import seaborn as sns


def create_directory(name='out', autoincrement=False, replace=True):
    '''Create a directory for logging results.

    name -- str -- base name

    replace -- bool -- if True, replace an existing directory

    autoincrement -- bool -- if True, append an autoincrementing number to the
    directory name

    returns -- str -- created directory
    '''
    if autoincrement:
        parent, fname = os.path.split(name)
        parent = '.' if parent == '' else parent
        pattern = re.compile(fname + r'(\d+)')
        if os.path.exists(parent):
            matches = [pattern.match(f) for f in os.listdir(parent)]
            n = max(
                (1 + int(m.group(1)) for m in matches if m is not None),
                default=0)
        else:
            n = 0
        name = os.path.join(parent, '{}{}'.format(fname, n))

    if os.path.exists(name):
        if replace:
            shutil.rmtree(name)
        else:
            raise FileExistsError(
                ('Directory {} already exists - pleas specify either'
                 ' autoincrement=True or replace=True').format(name))

    os.makedirs(name)
    return name


# Logging

class PrintLogger:
    def log(self, key, value):
        value = float(value.data if hasattr(value, 'data') else value)
        print('{}: {}'.format(key, value))


class JsonLogger:
    def __init__(self, path):
        self._log = open(path, 'w')
        self._t0 = time.time()

    @property
    def path(self):
        return self._log.name

    def log(self, key, value):
        value = float(value.data if hasattr(value, 'data') else value)
        self._log.write('{}\n'.format(
            json.dumps(dict(
                t=time.time() - self._t0,
                key=key,
                value=float(value)))))
        self._log.flush()


def load(f):
    with open('../out/log.jsonl') as f:
        return pd.DataFrame.from_dict([json.loads(line) for line in f])


def show(f):
    return sns.lmplot(
        data=load(f),
        x='t', y='value', row='key',
        fit_reg=False, size=5, aspect=2, markers='.', sharey=False)


# Monitoring

class Monitor(gym.wrappers.Monitor):
    '''Helper wrapper to make it easy to watch progress in stdout & in the
    log.
    '''
    def __init__(self, env, directory, print_every=10, **base_args):
        super().__init__(env, directory=directory, **base_args)
        self._print_every = print_every
        self._t0 = time.time()
        self._last_print = dict(time=self._t0, episode=0)
        self.logger = JsonLogger(os.path.join(self.directory, 'log.jsonl'))
        self._logged = 0

    def log(self, key, value):
        return self.logger.log(key, value)

    def _after_reset(self, observation):
        super()._after_reset(observation)

        lengths = self.get_episode_lengths()
        if self._logged < len(lengths):
            rewards = self.get_episode_rewards()
            for i in range(self._logged, len(lengths)):
                self.logger.log('episode.length', lengths[i])
            for i in range(self._logged, len(rewards)):
                self.logger.log('episode.reward', rewards[i])
            self._logged = len(lengths)

        now = time.time()
        if (self._print_every is not None and
                self._print_every < now - self._last_print['time']):
            e0 = self._last_print['episode']
            print(('t: {time:<5}'
                   '  episodes: {episodes:<7}'
                   '  av-reward: {reward:<4.3g}'
                   '  av-length: {steps:<4.3g}').format(
                       time='{:.0f} s'.format(now - self._t0),
                       episodes='{}-{}'.format(e0, self.episode_id),
                       reward=np.mean(self.get_episode_rewards()[e0:]),
                       steps=np.mean(self.get_episode_lengths()[e0:]),
                   ))
            self._last_print = dict(
                time=now, episode=len(self.get_episode_lengths()))
