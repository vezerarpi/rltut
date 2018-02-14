# rltut: A Deep Reinforcement Learning Tutorial

# Monday 19 Feb, 13:30 - 16:30, Brick Lane/Columbia Market

Teach computers to do things! With deep learning, we teach computers to approximate functions. With reinforcement learning we use functions to learn good behaviour of an agent acting in an environment (e.g. a player in a video game). Deep reinforcement learning combines these & works quite well (e.g. agents learn to play Atari video games at superhuman level).
No setup required, no prior reinforcement learning or deep learning knowledge required - just bring a laptop & be ready to do some programming! We'll be working in iPython, so if you're familiar with Python that'll help. We'll introduce the idea behind a reinforcement learning algorithm, then set you loose implementing it & training some agents to solve a simple task.

# Examples
## CartPole

We'll start with a simple game whose state is easy to visualise and hopefully is easy to solve for your program.

Iterations | 9 | 64 | 100 | 200
---|---|---|---|---
 | ![CartPole 9 Iterations](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/cartpole/openaigym.video.0.3537.video000009.gif | width=160) | ![CartPole 64 Iterations](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/cartpole/openaigym.video.0.3537.video000064.gif | width=160) | ![CartPole 100 Iterations](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/cartpole/openaigym.video.0.3537.video000100.gif | width=160) | ![CartPole 200 Iterations](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/cartpole/openaigym.video.0.3537.video000200.gif | width=160) |


## Further Work: Breakout
Once your program has mastered CartPole you could try an Atari game like Breakout.

This has been training on a laptop overnight, using a only a scaled down image of each frame (scaled to improve training time).

### 10 Games
![Breakout 10 Episodes](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/openaigym.video.0.36495.video000010.gif | width=160)

### 100 Games
![Breakout 100 Episodes](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/openaigym.video.0.36495.video000080.gif | width=160)

### 200 Games
![Breakout 200 Episodes](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/openaigym.video.0.36495.video000200.gif | width=160)

### 400 Games
![Breakout 400 Episodes](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/openaigym.video.0.36495.video000600.gif | width=160)

### 600 Games
![Breakout 600 Episodes](https://github.com/vezerarpi/rltut/blob/ideas-fair/assets/ideas-fair/openaigym.video.0.36495.video000580.gif | width=160)
