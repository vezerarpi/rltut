# coding: utf-8
import operator
import random
import scipy.misc
import signal
import time

import gym
import chainer as C
import multiprocessing as mp
import numpy as np


env_name = 'Breakout-v4'
exp_init_size = 500
n_episodes = 8000
# number of episodes after which to print q function evaluation
eval_period = 20
rng = random.Random()
rng.seed(42)


class Model(C.Chain):
    def __init__(self, observation_shape, n_actions):
        super().__init__()
        in_channels = observation_shape[2]
        out_channels1 = 8
        kernel_size1 = 8
        stride1 = 4
        out_channels2 = 16
        kernel_size2 = 4
        stride2 = 2
        projection_size = 128
        # Chain.init_scope is necessary for gradient book-keeping to be set up
        # for all the links defined below, otherwise errors are not
        # propagated back through the graph
        with self.init_scope():
            self.conv1 = C.links.Convolution2D(in_channels, out_channels1,
                                               kernel_size1, stride1,
                                               nobias=True)
            self.conv2 = C.links.Convolution2D(out_channels1, out_channels2,
                                               kernel_size2, stride2,
                                               nobias=True)
            self.projection = C.links.Linear(None, projection_size)
            self.action_values = C.links.Linear(projection_size, n_actions)

    def __call__(self, state):
        z1 = C.functions.relu(self.conv1(state))
        z2 = C.functions.relu(self.conv2(z1))
        z3 = C.functions.relu(self.projection(z2))
        return self.action_values(z3)


def print_params(chain):
    for path, param in chain.namedparams():
        print(path, param)


env = gym.make(env_name)
env.seed(0)


def scale_img(img):
    h = env.observation_space.shape[0] // 2
    w = env.observation_space.shape[1] // 2
    return scipy.misc.imresize(img, (h, w)).astype(dtype=np.float32)


def prepare_input(x):
    assert len(x.shape) in [3, 4]
    if len(x.shape) == 3:
        x = x.reshape((-1,) + x.shape)
    x = np.moveaxis(x, 3, 1).astype(np.float32)
    return x / 255.  # normalise pixel values


# New async part
class AsyncAgent:
    def model_params(self):
        raise NotImplementedError()

    def act(self, state):
         raise NotImplementedError()

    def step_update(self, reward, next_state):
        raise NotImplementedError()

    def episode_update(self):
        raise NotImplementedError


def setup_signal_handlers(shared_flag):
    def handler(num, frame):
        shared_flag.value = 1

    for sig in [signal.SIGABRT, signal.SIGINT,
                signal.SIGTERM, signal.SIGSTOP]:
        signal.signal(sig, handler)


def setup_shared_params(orig_params):

    shared_params = {}
    for key, param in orig_params.items():
        assert param.dtype == np.float32
        shared_params[key] = mp.RawArray('f', param.data.ravel())
    for key, param in orig_params.items():
        if key in shared_params:
            param.data = np.from_buffer(
                shared_params[key], dtype=param.data.dtype).reshape(param.data.shape)
            print('sharing param', key, param.data.shape)
    return shared_params


def train_async(agent_func, env_func, optim_func, n_procs):
    # init master agent
    master_agent = agent_func(0, optim_func())#XXX model needs to be forwarded to initialise all params
    # returns params in a dict
    master_params = setup_shared_params(master_agent.model_params())#XXX
    # move master params to shared arrays

    shared_stop_flag = mp.RawValue('B', 0)  # uint8, 0 for go, 1 for stop

    def set_stop_flag():
        shared_stop_flag.value = 1

    # create n new processes with env, agent wrapped in a runner func capturing the shared params
    def runner(agent_func, env_func, optim_func, stop_flag, process_idx):
        optim = optim_func()#XXX
        agent = agent_func(process_idx, optim)#XXX
        env = env_func()#XXX
        for ep in range(n_episodes):
            pull_from_shared(master_params, agent.model_params())#XXX
            state = env.reset()
            state = scale_img(state)
            done = False
            t_ep_start = time.time()
            while not done:
                if shared_stop_flag.value:
                    break
                state, reward, done, _ = env.step(agent.act(np.array(state, dtype=np.float32)))
                state = scale_img(state).astype(np.uint8)
                agent.step_update(reward, None if done else state)
            if shared_stop_flag.value:
                break
            agent.episode_update()
            push_to_shared(master_params, agent.model_params())#XXX
            t = time.time() - t_ep_start
            steps = env.get_episode_lengths()[-1]
            print('episode', ep, 'duration', t, 'steps', steps, 'steps/sec', steps / t)
            if ep % eval_period == 0:
                ep_lengths = env.get_episode_lengths()[-eval_period:]
                print('-' * 11)
                if ep:
                    print('episodes', ep - eval_period, '-', ep, 'steps', ' '.join(map(str, ep_lengths)))

    # start each proc and then join it
    setup_signal_handlers()  # flags all processes to halt
    procs = {}
    for process_idx in range(1, n_procs+1):
        procs[process_idx] = mp.Process(target=runner,
                                        args=(agent_func,
                                              env_func,
                                              optim_func,
                                              shared_stop_flag,
                                              process_idx))
    for p in procs.values():
        p.start()
    for process_idx, p in procs.items():
        p.join()
        if p.exitcode != 0:
            print(process_idx, 'exit code', p.exitcode)
