# Car Racing with various Reinforcement Learning Techniques
## Background
## Related Works
## Theory
All reinforcement learning algorithms center around the estimation of the value of a particular configuration. That is, how much discounted reward can I expect starting off from a particular state V(s), state-action pair Q(s,a), or state distribution J(θ) given a particular policy. These can be estimated in many ways, from one extreme of complete Monte-Carlo rollouts (low bias, high of variance), to another extreme of one step updates where we bootstrap the values of their direct descendants (high bias, low variance). That is, given the difference in expected reward between adjacent nodes and terminal state, there should be only 1 true graph of the true values that we can reconstruct.
### Double Deep Q Networks
### Proximal Policy Optimization
## Methods
### Data Preprecessing
### Action Space Shaping
### Reward Shaping
## Model Architecture
### Double Deep Q Networks
### Proximal Policy Optimization
## Results and Discussion
## Future Works
