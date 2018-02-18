'''
# Deep Reinforcement Learning Tutorial

See the `notes.ipynb` notebook for a brief overview of the theory behind Q-learning.
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

See the `guide.ipynb` notebook for more details on the action-value Q function and the loss used to train an agent to estimate it. We will implement the following algorithm to sample batches of experiences (transitions from the current state to a next state following an action) and compute a loss to minimise that will lead the model to learning better and better approximations of expected rewards of all possible actions from any given state.

"""
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
"""

## Agent

We will implement the algorithm by factoring most of the logic and calculations around states, rewqards and the loss to an `Agent` class. This class will manage the selection of actions, tracking the current experience and the updates of the parameters of Q-function model.
'''
from .examples import model


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

    def reward(self, reward, next_state):
        '''
        Takes the reward for the alt action and the resulting next_state, calculates the Q-learning loss and performs a parameter update on the model.

         - reward a float, the reward for the latest act()
         - next_state a np.array containing the observed next state resulting
         from the latest act()
         - return None
        '''
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
        # batch-shaped Variable using C.functions.stack.

        # TODO calculate the loss using chainer functions
        # See C.functions.mean_squared_error
        loss = # TODO see the pseudo code above or the guide for the loss

        # These two lines compute the gradient fo the calculations that we just
        # did and update the model using the optimiser
        loss.backward()
        self._optim.update()

        # The average loss for the batch is returned
        return np.asscalar(loss.data)


'''
## Training loop

'''
from .examples import log


n_episodes = 1000
# number of episodes after which to print q function evaluation
eval_period = 20

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, directory=output_path, force=True)
env.seed(0)
agent = Agent()

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


'''
## Further work

Create a new agent that uses the `example.AtariModel` to learn to play the game 'Breakout-v4'. This is an Atari game environment whose observations are the RGB pixels of the screen and has the following actions: 0: do nothing, 1: move left, 2: request ball, 3: move right.

This task is a lot more computationally intensive so the image should be scaled down to a quater of its size before being passed to the model. Even with this it will take hours before it starts to score a few points but you should be able to see improvements in the first couple of hundred episodes. Once you are confident that the agent is beginning to learn to play then you could start modifying you it to execute the model calculations and updates on GPU.
'''
