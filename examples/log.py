import os
import re
import time
import glob
import json
import shutil
import gym
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display


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


# Videos

def find_videos(path=''):
    '''Return a list of videos.
    '''
    return glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)


def video(path):
    '''Show a video from a full path (or find the latest video in a directory).

    path -- string -- path to mp4 file, or a parent directory
    '''
    if os.path.isdir(path):
        # Find the latest video
        videos = find_videos(path)
        if len(videos) == 0:
            raise ValueError('No videos found in directory "{path}"'.format(
                path=path))
        path = max(videos, key=os.path.getmtime)
    # subpath = os.path.relpath(os.path.abspath(path), os.getenv('PWD'))
    subpath = os.path.relpath(os.path.abspath(path), os.getcwd())
    return IPython.display.HTML("""
    <video src="{subpath}" controls autoplay></video>
    <p>{subpath}</p>
    """.format(subpath=subpath))


# Log recoding

class PrintLogger:
    '''A substitute for JsonLogger to log to terminal, useful for testing.
    '''
    def log(self, key, value):
        value = float(value.data if hasattr(value, 'data') else value)
        print('{}: {}'.format(key, value))


class JsonLogger:
    '''Logs scalar events to a simple jsonlines file.
    '''
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


# Log loading/plotting

def _strip_common(files):
    '''Returns 'files' with common prefix/suffix stripped.
    '''
    if len(files) <= 1:
        return files
    start = next(
        idx for idx, chs in enumerate(zip(*files))
        if any(ch != chs[0] for ch in chs))
    end = next(
        idx for idx, chs in enumerate(zip(*(f[::-1] for f in files)))
        if any(ch != chs[0] for ch in chs))
    return [f[start:] if end == 0 else f[start:-end] for f in files]


def load(*names_or_patterns, only_match=None):
    '''Load log file(s) as a Pandas DataFrame.

    names_or_patterns -- string -- path to file, or glob pattern

    only_match -- string or None -- regex to filter unique names by

    returns -- pd.DataFrame
    '''
    files = [f for p in names_or_patterns
             for f in glob.glob(p, recursive=True)]
    dfs = []
    for path, name in zip(files, _strip_common(files)):
        if only_match is None or re.search(only_match, name):
            with open(path) as f:
                dfs.append(pd.DataFrame.from_dict(
                    [json.loads(line) for line in f]))
                dfs[-1]['name'] = name
    return pd.concat(dfs)[['name', 'key', 't', 'value']]


def _autorange_y(ys, tail=0.05):
    '''Autorange the y axis based on the given ys.

    ys -- pd.Series

    tail -- float -- proportion of y to exclude

    returns -- (float, float) -- preferred range
    '''
    return (ys.quantile(tail, interpolation='lower'),
            ys.quantile(1 - tail, interpolation='higher'))


def plot(data, nsmooth=None, point_alpha=None, color=None, ax=None):
    '''Plot datapoints & smoothed curve for elements of data.

    data -- pd.DataFrame -- plots the 'value' column

    nsmooth -- int -- smoothing window to apply (or 1 to disable)

    point_alpha -- float -- opacity of points (or 0 to disable points)

    color -- tuple -- as from sns.color_palette

    ax -- plt.axes -- e.g. from gca()
    '''
    if ax is None:
        ax = plt.figure().gca()
    if nsmooth is None:
        nsmooth = int(len(data) / 50)
    if point_alpha is None:
        point_alpha = max(0.01, 1 / (1 + len(data) / 1000))
    if nsmooth <= 1:
        data.plot(x='t', y='value', color=color, ax=ax)
    else:
        if 0 < point_alpha:
            data.plot(x='t', y='value', marker='.', kind='scatter',
                      alpha=point_alpha, color=color, ax=ax)
        y = data['value'].rolling(
            min_periods=0, window=2 * nsmooth, win_type='gaussian'
        ).mean(std=nsmooth)
        plt.plot(data['t'], y, color=color)
        ax.set_ylim(_autorange_y(y, tail=0))
    plt.ylabel('')
    plt.xlabel('t')


def plot_many(data, colors=None, nsmooth=None, point_alpha=None, ax=None):
    '''Plot datapoints & smoothed curve for elements of data.

    data -- pd.DataFrame -- plots the 'value' column, grouped by 'name'

    colors -- dict(string -> tuple) -- mapping names to colors

    nsmooth -- int -- smoothing window to apply (or 1 to disable)

    point_alpha -- float -- opacity of points (or 0 to disable points)

    ax -- plt.axes -- e.g. from gca()
    '''
    if ax is None:
        ax = plt.figure().gca()
    names = sorted(set(data.name))
    for name in names:
        plot(data[data.name == name],
             nsmooth=nsmooth,
             point_alpha=point_alpha,
             color=None if colors is None else colors[name],
             ax=ax)
    ranges_min, ranges_max = zip(*list(
        _autorange_y(data[data.name == name].value)
        for name in names))
    ax.set_ylim((min(ranges_min), max(ranges_max)))
    plt.legend(names)


def show(*names_or_patterns, show_points=True, keys=None, logs=None):
    '''High level helper to display all traces from a log file.

    show_points -- bool -- draw semitransparent points?

    keys -- string -- pattern to match keys to display

    logs -- string -- pattern to match logs (by name) to display
    '''
    # Simpler version:
    # sns.lmplot(data=load(f), x='t', y='value', row='key',
    #            fit_reg=False, size=5, aspect=2, markers='.', sharey=False)
    data = load(*names_or_patterns, only_match=logs)
    keys = sorted(k for k in set(data.key)
                  if keys is None or re.search(keys, k))
    colors = {name: color
              for name, color in zip(
                      sorted(set(data.name)),
                      sns.color_palette('hls', len(set(data.name))))}
    plt.figure(figsize=(10, 6 * len(keys)))
    for row, key in enumerate(keys):
        plt.subplot(len(keys), 1, row + 1)
        plot_many(data[data.key == key],
                  point_alpha=(None if show_points else 0),
                  colors=colors,
                  ax=plt.gca())
        plt.ylabel(key)


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

        rewards = self.stats_recorder.episode_rewards
        if self._logged < len(rewards):
            TYPEMAP = dict(t='training', e='evaluation')
            types = self.stats_recorder.episode_types
            rewards = self.stats_recorder.episode_rewards
            for i in range(self._logged, len(rewards)):
                self.logger.log(TYPEMAP[types[i]] + '.reward', rewards[i])
            self._logged = len(rewards)

        now = time.time()
        if (self._print_every is not None and
                self._print_every < now - self._last_print['time']):
            e0 = self._last_print['episode']
            print(('t: {time:<5}'
                   '  episodes: {episodes:<7}'
                   '  av-reward: {reward:<4.3g}').format(
                       time='{:.0f} s'.format(now - self._t0),
                       episodes='{}-{}'.format(e0, self.episode_id),
                       reward=np.mean(self.get_episode_rewards()[e0:]),
                       steps=np.mean(self.get_episode_lengths()[e0:]),
                   ))
            self._last_print = dict(
                time=now, episode=len(self.get_episode_lengths()))
