# Car Racing with various Reinforcement Learning Techniques
## Background
## Related Works
## Theory
All reinforcement learning algorithms center around a value estimation of a particular configuration. That is, how much discounted reward can I expect starting off from a particular state V(s), state-action pair Q(s,a), or state distribution J(θ) given a particular policy. These can be estimated in many ways, from one extreme of complete Monte-Carlo rollouts (low bias, high variance), to another of one step updates we bootstrap the values off their direct descendants (high bias, low variance). That is, with only information on the difference between adjacent and terminal nodes, there should be only 1 true value graph that we can reconstruct.

Given these values optimal actions can be chosen by simply selecting the highest next confiugration, or by directly modifying the policy action function.
### Double Deep Q Networks
Q-learning is a type of one-step update involving the value estimation of state-action pairs Q(s,a). A defining feature of Q-learning is that our agent can learn off policy. That is, it can explore suboptimal actions, while still updating our Q(s,a) knowledgebase so that its consistent with the optimal action policy. This is possible because our sampled sub-optimal Q(s,a) pairs are bootstrapped from the optimal ensuing Q(s<sup>'</sup>,a) pair (see figure ?) instead of our agents actual trajectory. Sampled actions are stored in an action replay buffer, and updating is done by randomly sampling experiences every few steps to reduce correlations between training data and for increased stability.

With Deep Q Networks we use a neural network to predict the Q(s,a) value of encountered configurations. Training minimizes the loss between our predicted Q(s,a) and what its supposed to be if bootsrapped off its descendant (! diagram of training update). Double Deep Q Networks use 2 nearly identical networks to stabilize training by minimizing immidiate chasing of moving targets. By allowing the target R + γQ(s<sup>'</sup>,a) to be predicted by a fixed alternative network, updating the policy network does not immidiately shift the target and allows loss to temporarily converge. The target network is typically an older version of the policy network, with the frequency of its update being a hyperparememter to tune.

### Proximal Policy Optimization
Proximal Policy Optimization is a policy gradient method where actions are chosen according to a defined function f(s) = 𝜋(a|s). Our objective is to maximize the value of the policy J(θ), which the expected discounted reward from starting off in a state or state distribution given that policy. To update the policy we need the gradients, which the policy gradient theorom shows that it is proportional to the following equation. (! image of J(θ) gradient proportiaonality). All this equation states is that the gradient of the policy's value to its parameters, is proportional to a weighted mean of the gradient of the policies action probabilities. These action selection probabilities are weighted according to Q(s,a) (gradient is more sensitive to high value state-actions), and μ(s) (the distribution frequency of encountered states following that policy). Notice that there is a conflict of gradients going on here, in that it seems that increasing the probabiity of all actions is the correct solution. In reality, the 𝜋(a|s) probabilities for a state must add to one, so the Q(s,a) weighting factor will naturally increase and decrease the appropriate probabilites.

When training, we do not explicity calculate this equation, rather we assume that our sampled experiences will naturally follow the state distribution μ(s) in expectation. Therefore our update becomes (! equation of update). In training we only sample particular actions instead of a summation, so the equation becomes (! equation Q(s,a)\*gradprob/prob). In PPO we can replace Q(s,a) with the acutal return Gt (or bootstrapped), and we subtract a baseline to reduce the variance of updates (it can be proven that this does not affect the equality), where the baseline is a paramterized value estimate V(s). Inuitively this is quite simple. If we experience more return than expected, increase the probability of the action, otherwise decrease it. The (!CLIP) further encourages stability by limiting the change in probabilities after each update.

## Methods
### Data Preprecessing
|                      | Raw Data                      | Processed                 | Description |
|----------------------|-------------------------------|---------------------------|----------|
| Image Simplification | ![](images/unprocessed.png)   | ![](images/postprocessed.png) |   Greyscale the image and clip regions irrelevant to the model's task. This allows more efficient training by focusing compute power on key areas.      |
| Speed Extraction     | ![](images/speed_bar.png) |  speed magnitude ∈ {0,5}  | Extract speed bar and sum the pixels. Normalize range of speed. Value is discrete (6 observations) due to poor image resolution.         |
### Action Space Shaping
### Reward Shaping
## Model Architecture
### Double Deep Q Networks
### Proximal Policy Optimization
## Results and Discussion
## Future Works
