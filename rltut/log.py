import os
import re
import time
import numpy as np


def _open_log(root, name):
    '''Open a log file (text) in the given root directory,
    with an optional name.

    If name is specified, the file is called "{root}/{name}.log",
    which is overwritten if it already exists.

    If name is None, the file is given a unique incrementing integer
    name i.e. {root}/0.log, {root}/1.log, etc.
    '''
    if not os.path.isdir(root):
        os.makedirs(root)

    if name is not None:
        return open('{}/{}.log'.format(root, name))

    pattern = re.compile(r'(\d+)\.log')
    matches = [pattern.match(f) for f in os.listdir(root)
               if os.path.isfile(os.path.join(root, f))]
    next_log = max(
        (1 + int(m.group(1)) for m in matches if m is not None),
        default=0)
    return open('{}/{}.log'.format(root, next_log), 'w')


class Log:
    def __init__(self, env, print_every=15, root='logs', name=None):
        self._env = env
        self._print_every = print_every
        self._t0 = time.time()
        self._last_print = dict(time=self._t0, episode=env.episode_id)
        self._log = _open_log(root, name)

    def __repr__(self):
        return 'Log({})'.format(self._log.name)

    @property
    def name(self):
        return self._log.name

    def tick(self):
        now = time.time()
        if (self._print_every is not None and
                self._print_every < now - self._last_print['time']):
            e0 = self._last_print['episode']
            print(('t: {time:>3.0f} s'
                   '  episodes: {episodes:<5}'
                   '  av-reward: {reward:>5.3g}'
                   '  av-length: {steps:>5.3g}').format(
                       time=now - self._t0,
                       episodes='{}-{}'.format(e0, self._env.episode_id),
                       reward=np.mean(self._env.get_episode_rewards()[e0:]),
                       steps=np.mean(self._env.get_episode_lengths()[e0:]),
                   ))
            self._last_print = dict(time=now, episode=self._env.episode_id)
