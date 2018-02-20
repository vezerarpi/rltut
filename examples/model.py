import chainer as C
import numpy as np


class CartPoleModel(C.Chain):
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

    def print_eval(self):
        '''
        Evaluates the model at 7 pole angles between +/-10 degrees. The other values of the state are 0.0, so the cart is in the middle of the space and not moving and the pole currently has 0.0 velocity. Prints Q values and indicators showing which action has the highest value at each state.

        '''
        theta_limit = 10 * 2 * np.pi / 360
        n = 7
        thetas = np.flip(np.linspace(-theta_limit, theta_limit, n), axis=0)
        states = np.array([[0.0, 0.0, theta, 0.0]
                           for theta in thetas],
                          dtype=np.float32)
        print('Eval Theta', ''.join(['[{:^12.1f}]'.format(x)
                                    for x in states[:, 2] * 360 / 2 / np.pi]))
        qs = self(C.Variable(states)).data
        self.cleargrads()
        a = ['_R' if x else 'L_' for x in np.argmax(qs, axis=1)]
        print('Eval L - R', ''.join(['[{:^12.1f}]'.format(l - r)
                                    for l, r in qs]))
        print('Eval Q    ', ''.join(['[{:5.2f}{}{:5.2f}]'.format(l, y, r)
                                    for (l, r), y in zip(qs, a)]))



class AtariModel(C.Chain):
    '''
    A small, 2 layer convolutiuonal model for learning to play Atari games,
    e.g. 'Breakout-v4'. See 'Asynchronous Methods for Deep REinforcement Learning
    (https://arxiv.org/abs/1602.01783) for details.
    '''
    def __init__(self, observation_shape, n_actions):
        super().__init__()
        in_channels = observation_shape[2]
        # The convolution channels and projection size are halved here compared
        # to the experimants in https://arxiv.org/abs/1602.01783 in order to be
        # fast enough to train on a CPU in reasonable time (still slow though)
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
