'''
# Deep Reinforcement Learning Tutorial

An introduction to reinforcement learning that creates an agent that learns to play the CartPole environment in OpenAI Gym using a slightly simplified version of the DQN algorithm.

See the `Guide.ipynb` notebook for a brief introduction to the theory behind Q-learning.
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

from .examples import model, log


# TODO replace the output path with your name
output_path = 'out/REPLACE-ME'
# create a random number generator and seed it so runs are repeatable
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
## Logging and video output

Gym can output videos of episodes, which for CartPole can be played in real time. This is useful for seeing how well an algorithm has learnt to play the game. This is easily done by wrapping the environment in a `gym.wrappers.Monitor`, which has the following constructor arguments:
* `directory` - output directory path, make sure you've personalised it above
* `force` - flag to force creation of new, or overwriting of existsing output directires
* `video_callable` - a function that takes the episode number and returns whther to record a video, defaults to every cubic number or every 1K episodes after the first 1K

The `examples.log` module has a Monitor class wrapping gym's Monitor to provide extra logging at the end of each episode as well as recording videos, both of which can be viewed in the `examples.Log.ipynb` notebook. The `log.Monitor` class accepts all the aarguments mentioned above.
'''
env = log.Monitor(env, directory=output_path, print_every=1,
                  force=True, video_callable=lambda ep: ep % 10 == 0)
# TODO Try the previous random actions here again but now with the monitor wrapping the env
# TODO and view the output by opening  and running the Log notebook.

'''
## Q-learning

See the `Guide.ipynb` notebook for more details on the action-value Q function and the loss used to train an agent to estimate it. We will implement the following algorithm to sample batches of experiences (transitions from the current state to a next state following an action) and compute a loss to minimise that will lead the model to learning better and better approximations of expected rewards of the possible actions at any given state.

```
Initialise replay buffer D
Initialise Q function with random weights w
for episode = 1, M:
    Initialise the environment and the initial state
    for t = 1, T:
        With probability eps selct a random action a_t
        otherwise select a_t = argmax(Q(state)) over possible actions a
        Execute action a_t in the environment and observe next_state, r_t, done
        Store (state, a_t, r_t, next_state, done) in D
        Sample a batch_size batch of experiences (s_j, a_j, r_j, s_j+1, done) from D
        Set loss = 0
        For each experience in batch:
            set y = r_j if done else
                    r_j + gamma * max(Q(s_j+1))
            loss += (y_j - Q(s_j)[a_j])**2
        loss /= batch_size
        Perform a gradient update step on the loss wrt the weights w
        if done: break
```

## Agent

We will implement the algorithm by factoring most of the logic and calculations around states, rewqards and the loss to an `Agent` class. This class will manage the selection of actions, tracking the current experience and the updates of the parameters of Q-function model.
'''


class Agent:
    def __init__(self):
        '''
        Create an agent containing
         * a model
         * an optimiser
         * an experience buffer
        '''
        self._lr = 0.01
        self._model = model.CartPoleModel()
        # This sets up a chainer optimiser that will be used to backprop the
        # loss through the model
        self._optim = C.optimizers.SGD(lr=self._lr)
        self._optim.setup(self._model)
        # TODO create a member for the experience replay buffer (just store in a
        # list).

    def act(self, state):
        '''
        Returns an action (as an int) for the current state. Keeps track of the state and the action taken for use in the next call to reward(). The action is chosen e-greedily (a random action is taken with probability e).

         - state a np.array representing the current observed state
         - return the selected action
        '''
        # Wrap the state in a chainer.Variable ready to be passed as an argument
        # to the model. This is how data must be fed in to Chainer models. The
        # value of a Variable can be accessed as a numpy array using its .data
        # property. The state is reshaped to have an outer array dimension as
        # chainer requires all inputs to have a batch dimension. In this case we
        # have a batch of 1 x state with shape (1, 4).
        model_input = C.Variable(state.reshape((1, -1)))

        # TODO Store the state that was passed in

        # TODO Select and store an action using self._model or choosing a random
        # action with a small probability. Return the action.

    def reward(self, reward, next_state, done):
        '''
        Takes the reward for the last action and the resulting next_state,
        calculates the Q-learning loss and performs a parameter update on the
        model for a miniubatch sampled from the experience buffer.

          - reward a float, the reward for the latest act()
          - next_state a np.array containing the observed next state resulting
         from the latest act()
          - done a bool indicating whether the next state is a terminal state
          - return The average loss for the latest batch
        '''
        # TODO Append the latest experience to the replay buffer. The experience
        # should contain (current state, action, reward, next state, done),
        # where current state and action should have been stored by the agent on
        # the last call to act(). It might help to store each experience as a
        # dict so that the lookups from it are easily readable.

        batch_size = 64
        # TODO Sample a batch from your replay buffer. See
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.sample.html

        # TODO Evaluate the next states in the sampled batch. These can be done
        # in a loop or all at once if you use np.stack to concatenate all you
        # next states along an outer batch dimension. Remeber that a batch
        # dimension is always needed by the model if evaluating the states
        # individually.
        y = # TODO the output of the model on the next state(s)

        # Importantly, the Q value results for the next states must be converted
        # back to a numpy array before being used in subsequent calculations.
        # This is because we do not want gradients to be applied as a result of
        # the previous calc, as we want the loss to move the Q values for the
        # current state closer to the values of the next states, leaving hte
        # next state values unchanged. Converting to numpy breaks the chain of
        # gradient bookkeeping in Chainer.
        y = y.data

        # This is required to make sure only the current calculations' gradients
        # are used in the update when we call loss.backward(). All loss
        # calculations, except the y values mentioned above need to come AFTER
        # this cleargrads() call.
        self._model.cleargrads()

        # Calculate the loss.

        # TODO Get the q values for the current states in the sampled batch from
        # the model, keep these as Chainer Variables by making sure you do not
        # call data on them and use Chainer operations to implement the loss.

        # TODO With the Q values for all actions at each state in the batch
        # calculated, we need to select the Qs for the actual action that was
        # taken for each state, according to the sampled experiences. The
        # Chainer Variable that holds these values can be indexed like a numpy
        # array, or the actions can be selected individually and combined into a
        # batch-shaped Variable using C.functions.stack. See
        # http://docs.chainer.org/en/stable/reference/generated/chainer.functions.stack.html

        # TODO Calculate the loss using chainer functions. Remember to take the
        # done flag for each element of the batch into account.
        # See C.functions.mean_squared_error
        # http://docs.chainer.org/en/stable/reference/generated/chainer.functions.mean_squared_error.html
        loss = # TODO see the pseudo code above or the guide for the loss

        # These two lines compute the gradient fo the calculations that we just
        # did and update the model using the optimiser
        loss.backward()
        self._optim.update()

        # The average loss for the batch is returned
        return np.asscalar(loss.data)


'''
## Training loop

Now that we have an agent we can start training it over multiple episodes of the environment.

It might be useful to print out an evaluation of the model for a fixed set of states to be able to check that the values are changing and that it is beginning to behave as expected in those states.

After a certain number of episodes you can use the `eval_cartpole` function to print the Q values form the model for 7 pole angles between +/-10 degrees. These Q values should increase over time as your model experiences longer episodes , and thus more rewards, when it chooses actions that help stabilise the pole. If your values are not changing from one iteration to the next then there may be a problem with the chainer code that is stopping gradient updates s from being applied back through the components of the model. These outputs are only a rough indicator of how well your model is doing. It is unlikely to ever see exactly these states so it may not have made the best decision for each of them, but if the rewards are not increasing and the Q values do not look like they are moving towards values that you would expect given the pole orientation then something is probably wrong.
'''


def eval_cartpole(agent):
    '''
    Evaluates the agent's Q-function at 7 pole angles between +/-10 degrees. The other values of the state are 0.0, so the cart is in the middle of the space and not moving and the pole currently has 0.0 velocity. Also prints indicators showing which action has the highest value at each state.

    Expects agent to have a _model member that is the chainer model for its Q function.
    '''
    theta_limit = 10 * 2 * np.pi / 360
    n = 7
    thetas = np.flip(np.linspace(-theta_limit, theta_limit, n), axis=0)
    states = np.array([[0.0, 0.0, theta, 0.0]
                       for theta in thetas],
                      dtype=np.float32)
    print('Eval Theta', ''.join(['[{:^12.1f}]'.format(x)
                                 for x in states[:, 2] * 360 / 2 / np.pi]))
    qs = agent._model(C.Variable(states)).data
    agent._model.cleargrads()
    a = ['_R' if x else 'L_' for x in np.argmax(qs, axis=1)]
    print('Eval L - R', ''.join(['[{:^12.1f}]'.format(l - r)
                                 for l, r in qs]))
    print('Eval Q    ', ''.join(['[{:5.2f}{}{:5.2f}]'.format(l, y, r)
                                 for (l, r), y in zip(qs, a)]))


n_episodes = 1000
# Re-initialise the Environment and Monitor
env = gym.make('CartPole-v0')
env = log.Monitor(env, directory=output_path, print_every=1,
                  force=True, video_callable=lambda ep: ep % 10 == 0)
env.seed(0)
agent = Agent()

for ep in range(n_episodes):
    # TODO step through the environment until done, using agent.act() and
    # agent.reward()

# TODO print the total reward after each episode, optionally also call
# eval_cartpole() to see how the agent is learnign to estimate those particular
# states every now and then (e.g. every 20 episodes)

'''
## Further work

Create a new agent that uses the `example.AtariModel` to learn to play the game 'Breakout-v4'. This is an Atari game environment whose observations are the RGB pixels of the screen and has the following actions: 0: do nothing, 1: move left, 2: request ball, 3: move right.

This task is a lot more computationally intensive so the image should be scaled down to a quater of its size before being passed to the model. Even with this it will take hours before it starts to score a few points but you should be able to see improvements in the first couple of hundred episodes. Once you are confident that the agent is beginning to learn to play then you could start modifying you it to execute the model calculations and updates on GPU.
'''
