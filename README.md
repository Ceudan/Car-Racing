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
First, we greyscale and clip regions of the input that have low correlations to performance or are overly complex. This increases training efficiency by focusing more compute on key areas. Second, we extract speed information by summing the pixels of the speed bar and normalizing. We consider this more computationally efficient that passing stacked consecutive frames to the CNN as done by others. Note that speed magnitude is discrete due to image resolution limits.

|                      | Raw Data                    | Processed                     |
|----------------------|-----------------------------|-------------------------------|
| Image Simplification | ![](images/unprocessed.png) | ![](images/postprocessed.png) |
| Speed Extraction     | ![](images/speed_bar.png)   |speed magnitude ∈ {0,1,2,3,4,5} |
 
### Action Space
|          | Steering Angle | Throttle                                        | Brake   |
|----------|----------------|-------------------------------------------------|---------|
| Standard | ∈ [-1,1]       | ∈ [0,1]                                         | ∈ [0,1] |
| Modified | ∈ {-0.3,0,0.3}   | = 0.1 if speed<threshold<br/> &nbsp; &nbsp; &nbsp; 0 if speed>=threshold  | = 0     |

### Reward Shaping
To prevent the accumulation of poor quality experiences, we terminate episodes once a 0.1 second interval is spent on the grass and return a -100 reward. Since we fixed speed the learning task of our model focuses on staying on track.
## Model Architecture
Architecture was inspired by (! DDQN paper). Though our action space is simpler, we kept original dimensions for ease of comparisons.
### Double Deep Q Networks
Input = preprocessed image<br/>Output = Q values of steering actions
| Input Shape | Function                |
|-------------|--------------------------------|
| (96,84,1)   | Conv2d + LeakyRelu + Batchnorm |
| (20,23,8)   | Max-Pooling                    |
| (10,11,8)   | Conv2d + LeakyRelu + Batchnorm |
| (12,13,16)  | Max-Pooling                    |
| (6,6,16)    | LeakyRelu + Flatten            |
| (576+1)     | speed appended to tensor       |
| (256)       | LeakyRelu                      |
| (50)        | LeakyRelu                      |
| (3)         | Identity                       |
### Proximal Policy Optimization
Input = preprocessed image<br/>Output = beta distribution parameters and state value<br/> Note that the fully connected layers diverge.
| Input Shape | Function                |
|-------------|--------------------------------|
| (96,84,1)   | Conv2d + LeakyRelu + Batchnorm |
| (20,23,8)   | Max-Pooling                    |
| (10,11,8)   | Conv2d + LeakyRelu + Batchnorm |
| (12,13,16)  | Max-Pooling                    |
| (6,6,16)    | LeakyRelu + Flatten            |
| (576+1)     | speed appended to tensor       |
| (577)       | LeakyRelu                      |
| (577), &nbsp; &nbsp; &nbsp; &nbsp; (1)| LeakyRelu, &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Identity|
| (2)         | Softplus                       |
## Results and Discussion
### Double Deep Q Networks
Our best model obtained an average reward of 850/900 to officially solve the environment. It was able to visit 97% of track tiles travelling at moderate speeds.

We also tested various hyperparameters such as reward decay factor gamma, time discritization length, and speed at identical epislon and learning rate schedules. Simpler hyperparamter settings yielded the best performance. This corresponds to higher reward decays, and longer discritization lengths (! top left of table). It is possible that our short training sessions (30 minutes/360 episodes) was not sufficient to capitalize on the greater expressive power of more complex hyperparemeters. With regards to speed,  we observe that the track is easily mastered at moderate speeds, almost never completed at fast speeds, and hardly makes progress at very fast speeds. Specifically, the car would skid out at turns and contact with grass. We infered that the environment designers implied for dynamic braking and acceleration policy to be required in the winning solution.
### Proximal Polixy Optimization
## Future Works
