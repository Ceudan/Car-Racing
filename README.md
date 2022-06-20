# Car Racing with various Reinforcement Learning Techniques
## Background
The purpose of this project was to become familiar with reinforcement learning libraries, algorithms and hyperparameter tuning. To accomplish this I hand coded the Double Deep Q Learning and Proximal Policy Approximation algorithms for OpenAI Gym's Car Racing Environment (environment details here [[1]](https://www.gymlibrary.ml/environments/box2d/car_racing/)). Interestingly I noticed all top implementations required extremely high computational costs. Therefore my research question became, can I achieve similar results and beat the environment at significantly lower computational costs.
## Related Works
Below are top implementations from OpenAI Gym's officially leaderboard that use similar algorithms to ours [[2]](https://github.com/openai/gym/wiki/Leaderboard)).
| Algorithm    | Training Time              |
|--------------|----------------------------|
| PPO          | ≈10,000 episodes<br/>(5*10<sup>6</sup> environment steps)  |
| DQN          | 900 episodes               |
| World Models | 10,000+ episodes           |
| DQN          | ≈3000 episodes<br/>(1.6*10<sup>6</sup> environment steps) |

Less costly implementations include PPO, DQN, AC3 models done by Stanford Students which never surpassed a score of 600 [[3]](https://pdfcoffee.com/reinforcement-car-racing-with-a3c-pdf-free.html), and another PPO which achieved a true score of 820 with 2700 episodes [[4]](https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/CarRacing-From-Pixels-PPO).
## Theory
All reinforcement learning algorithms center around a value estimation of a particular configuration. That is, how much discounted reward can I expect starting off from a particular state V(s), state-action pair Q(s,a), or state distribution J(θ) given a particular policy. These can be estimated in many ways, from one extreme of complete Monte-Carlo rollouts (zero bias, high variance), to the other of one-step bootstrap updates. That is with only information on the difference between adjacent and terminal nodes, there should be only 1 true value graph that we can reconstruct. Optimal actions are chosen by simply selecting the highest next configuration, or by directly modifying an action policy function.
### Double Deep Q Networks
Q-learning estimates the value of state-action pairs Q(s,a) using one-step updates. A defining feature of Q-learning is that our agent can learn off policy. That is, it can explore suboptimal actions, while still updating our Q(s,a) knowledgebase so that its consistent with the optimal action policy. This is possible because our free to choose sampled Q(s,a) pairs are bootstrapped off the optimal ensuing Q(s<sup>'</sup>,a) pair (see figure ?), regardless of our agents actual trajectory. Memory is stored in action replay buffers, with experiences being randomly selected at training time to reduce inter correlations between data increasing training stability.

![visualization of Q-learning update](images/Q-update.PNG)

Figure 1: Visualization of Q-update (* denotes optimal actions).
&nbsp;

Deep Q Networks we use a neural network to predict the Q(s,a) value of encountered states. Training minimizes the loss between our predicted Q(s,a) and what its theoretical value if bootsrapped off its descendant (! diagram of training update). Double Deep Q Networks stabilize training by allowing the target R + γQ(s<sup>'</sup>,a) to be predicted by an alternative nearly identical network. This avoids immidiate chasing of moving targets where updating the policy network does not immidiately shift the target and allows loss to temporarily converge. The frequency to which to update the target network is a hyperparameter to tune.

### Proximal Policy Optimization
Policy Gradient methods sample actions according to a probability distribution. The parameters of the probability distribution can be output by a neural network. Our objective is to maximize the value of the policy J(θ), which is the expected discounted reward from starting off in a state distribution given that policy. To update the policy we need the gradients, which is shown in figure 2. All this equation states is that the gradient of the policy's value to its parameters, is proportional to a weighted mean of the gradient of the policy's actions probabilities. These action probabilities are weighted according to Q(s,a) (J(θ) is more sensitive to high valued actions), and μ(s) (the distribution frequency of encountered states for that policy). Notice that it seems that increasing the probabiity of all actions is the correct solution. In reality, the 𝜋(a|s) probabilities for a state must add to one, so the Q(s,a) weighting factor will naturally increase and decrease the appropriate probabilites. When training, we do not explicity calculate μ(s), rather we assume that our sampled experiences will naturally follow the state distribution μ(s) in expectation. We also only sample a particular action per state visit instead of the summation, and we introduce the denominator to reweigh disproportionally sampled actions. Finally, we can replace Q(s,a) with real sampled reward returns Gt.

![Policy Gradient Equation](images/Policy_Gradient_Equation.png)

Proximal Policy Optimization is a type of actor-critic twist on the policy gradient theory. The actor network outputs parameters of the action policy as discussed above. The critic network outputs its value estimation of states which is used in updating. Looking at our policy loss we can see that we subtracted our critic's value estimation from our exprienced return Gt. This advantage function stabilizes updates by standardizing the feedback signal (it can be shown that subtracting a baseline does not affect the gradient proportionality). Inuitively, if Gt-V(s) is positive, we experienced more reward than expected, so we increase the probability of recent actions and vice versa. The critic network gets updated by the critic loss. Often the actor and critic share layers, so the terms are combined. The entropy of the policy distribution is added to discourage it from collapsing and encourage exploration. Finally we increase stability by wrapping the policy loss with the CLIP function which bounds the magnitude of single updates. The original PPO paper can be found here [[2]](https://arxiv.org/pdf/1707.06347.pdf).

![Proximal Policy Optimization Equation](images/PPO_equation.png)

We used the beta distribution for our policy, which matches well with our 1 dimensionsal bounded action space.

![Graph of Beta Distribution](images/Beta_Distribution.png)

## Methods
To reduce required training, we need to simplify the learnable relationship. We accomplished this by simplifying the input image, using a novel technique to extract speed information, and reducing the action space by deterministically controlling speed (agent need only control direction).
### Data Processing
First, we greyscale and clip regions of the input that have low correlations to performance or are overly complex. This increases training efficiency by focusing more compute on key areas. Second, we extract speed information by summing the pixels of the speed bar and normalizing. Extracting speed this way is more computationally efficient that passing stacked consecutive frames to the CNN as done by others. However information is lost since speed magnitude precision is limited by image resolution.

|                      | Raw Data                    | Processed                     |
|----------------------|-----------------------------|-------------------------------|
| Image Simplification | ![](images/unprocessed.png)<br/> &nbsp; &nbsp; &nbsp; (96,96,3) | ![](images/postprocessed.png)<br/> &nbsp; &nbsp; &nbsp; (84,96,1) |
| Speed Extraction     | ![](images/speed_bar.png)   |speed magnitude ∈ {0,1,2,3,4,5} |
 
### Action Space
|          | Steering Angle | Throttle                                        | Brake   |
|----------|----------------|-------------------------------------------------|---------|
| Standard | ∈ [-1,1]       | ∈ [0,1]                                         | ∈ [0,1] |
| Modified | ∈ {-0.3,0,0.3}   | = 0.1 if speed<threshold<br/> &nbsp; &nbsp; &nbsp; 0 if speed>=threshold  | = 0     |

### Reward Shaping
To prevent the accumulation of poor quality experiences, we terminate episodes once a 0.1 second interval is spent on the grass and return a -100 reward. This relatively high negative reward should make the model focus on simply staying on track (since speed is fixed).
## Model Architecture
DQN and PPO architectures were inspired by [[5]](https://github.com/jperod/AI-self-driving-race-car-Deep-Reinforcement-Learning/blob/master/SI_Final_Project.pdf) and [[6]](https://arxiv.org/pdf/2111.02202.pdf) respectively. Our models were likely overparamaterized due to our reduced action space. However we kept their structure for ease of comparison.
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
We achieved our goal of obtaining a high score at a lower training costs than all other observed implementations (850/900 score at 360 episodes). However our models were not able to break this ceiling and could not officially solve the environment (900 score). In future steps we propose interesting improvements that could help beat the environment while still maintaing our objective of lower compuatational costs.

(PPO...)
### Double Deep Q Networks
Our best model obtained an average reward of 850/900 to officially solve the environment. It was able to visit 97% of track tiles travelling at moderate speeds.

We also tested various hyperparameters such as reward decay factor gamma, time discritization length, and speed at identical epislon and learning rate schedules. Simpler hyperparamter settings yielded the best performance. This corresponds to higher reward decays, and longer discritization lengths (! top left of table). It is possible that our short training sessions (30 minutes/360 episodes) was not sufficient to capitalize on the greater expressive power of more complex hyperparemeters. With regards to speed,  we observe that the track is easily mastered at moderate speeds, almost never completed at fast speeds, and hardly makes progress at very fast speeds. Specifically, the car would skid out at turns and contact with grass. We infered that the environment designers implied for dynamic braking and acceleration policy to be required in the winning solution.
### Proximal Polixy Optimization
...
## Future Works
