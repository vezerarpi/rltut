# Deep Reinforcement Learning Tutorial

## Introduction

### Types of ML
#### Supervised and unsupervised learning
The loss or objective function that we want to optimise for is known in both of these settings. For supervised learning we have a dataset of *(input, output)* pairs, our models make predictions for the output values based on the inputs and then a loss function scores how good a prediction this was compared to the *correct* output.

Similarly for unsupervised learning, while there isn't a predefined output label for each input, there is a known objective function tht scores how well a model can describe the data. E.g. for language modelling there is a loss function (softmax cross-entropy) scoring how well the model can predict each word in the dataset from the words that came before it.

#### Reinforcement learning
Unlike the previous two scenarios, there isn'st a known objective function that we can directly optimise for. RL problems are usually framed as an agent interacting with an environment. The agent observes states, takes actions and recieves rewards from the environment but the inner workings or the true dynamics driving the environment cannot be directly observed.

The agent observes the current state of the environment and chooses and action, in response to which the environment provides a reward and the next state taht result form the action. RL algorithms try to maximise the rewards for the agent by learning better and better choices of actions.

Both the agent and environment can be stochastic.
