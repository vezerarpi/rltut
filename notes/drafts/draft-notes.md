# Deep Reinforcement Learning Tutorial

## Introduction

### Types of ML
#### Supervised and unsupervised learning
The loss or objective function that we want to optimise for is known in both of these settings. For supervised learning we have a dataset of *(input, output)* pairs, our models make predictions for the output values based on the inputs and then a loss function scores how good a prediction this was compared to the *correct* output.

Similarly for unsupervised learning, while there isn't a predefined output label for each input, there is a known objective function that scores some derived property of the data. E.g. word embeddings are trained to minimise the distance between the vectors representing similar words, or for language modelling there is a loss function (softmax cross-entropy) scoring how well the model can predict each word in the dataset from the words that came before it.

#### Reinforcement learning
Unlike the previous two scenarios, there isn't a known objective function that we can directly optimise for. RL problems are usually framed as an agent interacting with an environment. The agent observes states, takes actions and recieves rewards from the environment but the inner workings or the true dynamics driving the environment cannot be directly observed.

The agent observes the current state of the environment and chooses and action, in response to which the environment provides a reward and the next state that result from the action. RL algorithms try to maximise the rewards for the agent by learning better and better choices of actions.

Both the agent and environment can be stochastic.

## The environment, states and actions

The environment is generally regarded as a black box in RL. As mentioned above the agent only observes the state $s_t$ returned by the environment and chooses and action $a_t$ to perform from among a pre-defined set of actions. After taking an action the environment is updated and returns the next state $s_{t+1}$ and a reward value $r_t$. This is repeated until the environment reaches a terminal state, at which point the environment is reset to some inital state and the process starts again. A series of steps from inital state to terminal state is called an episode.

## Rewards, policies and value functions

### Rewards and returns

The agent recieves a reward value after each step in the environment. The total return at a time $t$ is $R_t = \sum^\infty_{k=0} \gamma^k r_{t+k}$, the discounted sum of all future rewards using a discount factor $\gamma$ (e.g. 0.99). This way the return at each time step captures the rewards from subsequent actions and by keeping track of these an agent can choose between actions by considering which one it expects to get the best return for.

### Policies

A policy $\pi(s)$ is the probability of taking an action $a$ at state $s$, $\pi(s) = p(a|s)$. Some RL methods explicitly represent the policy and improve it directly but Q-learning uses what is called a Q-function that calculates the expected rewards of states and actions and only implicitly represents the notion of a policy.

### Experiences

As the agent ineracts with the environment it creates a sequence of states, actions and rewards, referred to as a trajectory $[s_0, a_0, r_0, s_1, a_1, r_1, ..., s_\$]$. Most RL algorithms work on subsequences of trajectories, for example the tuple $(s,a,r,s')$ covering a current state, chosen action, observed reward and the next state that resulted. These experiences could cover multiple time steps but here we will only consider a single step and its resulting next state.

### The Q function
For a given policy, the action-value function $Q(s, a)$ represents the value of all future rewards from taking action $a$ at state $s$ and then continuing to the follow the policy.
\begin{align}
Q^\pi(s_t,a_t) & = \mathbb{E}\bigl[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... \big|  \pi \bigr]\\
& = \mathbb{E}\bigl[r_t + \gamma \max_{a'}Q^\pi(s_{t+1},a') \big|  \pi \bigr]
\end{align}

$Q$ is the agent's best current estimate of the value of an action at the current state and can be recursively defined as the expected value of the current reward plus the discounted estimate of the best next action at the next state (resulting from the chosen action $a$). This function can be learnt iteratively by repeatedly playing the game to sample many $(s, a, r, s')$ experiences and then using those experiences to improve the estimates of $Q$.

The policy $\pi$ can be extracted from $Q$ by always taking the action with the maximal expected value at each time step.

### Q-learning loss

In Q-learning the loss is an expectation over $(s,a,r,s')$ experiences of the squared error between the current value estimate and the maximal value estimate at the next state.
\begin{equation*}
L_i(\theta_i) = \mathbb{E}_{(s,a,r,s')}\bigl[\bigl(r + \gamma \max_{a'}Q(s',a') - Q(s, a)\bigr)^2\bigr]
\end{equation*}

The Q function can be approximated by a neural network, which is usually configured to output the estimated value for all actions at the current state in one go: `action_values = model(curent_state)`. This neural network is updated using the loss function above. The $Q$ estimates improve as more and more experiences are sampled. This is because after a very strong reward signal has been experienced at $(s', a^\text{bad})$, the model will have knowledge of the bad value of $Q(s', a^\text{bad})$ the next time it experiences $(s,a,r,s')$, and this information therefore will be propagated backwards.

A neural network can be well suited to this sort of task as it can learn a representation of the states of the environment in order to estimate the Q values for them. The state space can have a very high dimension and be complex, even for a simple Atari game, but a good, distributed representation can find similarioties between the many possible states which makes it tractable for the model to learn to associate these with Q values for actions. The model's representation of the states it sees should work analogously to word vectors in word2vec, where words that are used similarly have similar embeddings. In this way the model doesn't need to remember each individual possible state but learn a fuzzy correlation between states and Q values.

### Experience replay

If we just use the current experience $(s,a,r,s')$ to update our Q function then it could suffer from overfitting to just the local parts of the state space that the agent is currently in, as each experience will be highly correlated to those around it. The DQN paper introduced the idea of an experience replay buffer to try to overcome this by calculating each Q function update based on a large number of experiences at once. Each experience the agent observes while playing the game is added to the buffer and instead of calculating the loss for the current experience, a *batch* of experiences sampled from the buffer is used instead. The function is updated using the average loss across all examples in the batch.

## Simplified DQN algorithm

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

## Further reading

* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf), Sutton and Barto, 2014. A text book covering almost all aspects of RL theory.
* [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/), Mnih et al., 2015. The original paper that used deep convolutional networks to approximate the Q function and learn to play Atari games.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih et al., 2016. An alternative to Q-learning, this paper also includes lots of information on Q learning as well as introducing an asynchronous version of Actor-Critic, a powerful alternative to DQN.
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel et al., 2017. This paper throws together all the mosts promising advacnes since DQN and evaluates them. Also includes good introductions to DQN and to the subsequent advances.
