import chainer as C


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
