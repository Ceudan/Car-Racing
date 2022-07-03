from utils import*

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
import numpy as np
import copy
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from torch import randint
from time import sleep

import os  


class DQN_Network(nn.Module):
  def __init__(self,gamma = None):
    super().__init__()
    #layers
    self.LeakyReLU = nn.LeakyReLU()
    self.conv1 = nn.Conv2d(1,8,kernel_size = 7, stride = 4,padding = 0)
    self.conv2 = nn.Conv2d(8,16,kernel_size = 3, stride = 1,padding = 2)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(577,256)
    self.fc2 = nn.Linear(256,50)
    self.fc3 = nn.Linear(50,3)
    self.batchnormCNN1 = nn.BatchNorm2d(num_features = 8)
    self.batchnormCNN2 = nn.BatchNorm2d(num_features = 16)
    self.batchnormFC1 = nn.BatchNorm1d(num_features = 256)
    self.flatten = nn.Flatten()
    self.gamma = gamma
  def forward(self,x):
    # reformat image (input = BS,96,96, or 96,96) (output = BS,1,96,96)
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    if(x.dim()==2):
      x = torch.unsqueeze(x,dim=0)
      x = torch.unsqueeze(x,dim=0)
    elif(x.dim()==3):
      x = torch.unsqueeze(x,dim=1)
    subimage = (x[:,:,84:96,13:14]-0.495)*10
    speed = torch.sum(subimage,dim=(2,3))
    x = x[:,:,:84,:]
    #plot_image(np.squeeze(x.detach().numpy()))
    
    #print(x.shape)
    x = self.batchnormCNN1(self.LeakyReLU(self.conv1(x)))
    #print(x.shape)
    x = self.pool(x)
    #print(x.shape)
    x = self.batchnormCNN2(self.LeakyReLU(self.conv2(x)))
    #print(x.shape)
    x = self.pool(x)
    #print(x.shape)
    x = self.flatten(x)
    #print(x.shape)
    x = torch.cat((x,speed),dim=1)
    x = self.batchnormFC1(self.LeakyReLU(self.fc1(x)))
    #x = self.LeakyReLU(self.fc1(x))
    #print(x.shape)
    x = self.LeakyReLU(self.fc2(x))
    #print(x.shape)
    x = self.fc3(x) 
    #print(x.shape)
    return x
  def get_action(self,state):
    qvals = self.forward(state)
    return torch.argmax(qvals,1) 
  def convert_action(self,action,state):
    # determine if you are going too fast
    speed = get_speed(state).item()
    if(speed>3.5):
      accel = 0
    elif(speed>2.5):
      accel = 0
    else:
      accel = 0.1
    # convert action from index, to a list of turning,engine,breaking strengths
    action = action.item()
    # Discretized action space (left-forward,straight-forward,right-forward)
    if(action == 0):
      return [-0.3,accel,0]
    elif(action == 1):
      return [0,accel,0]
    elif(action == 2):
      return [0.3,accel,0]
  



class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.steer_actions = []
        self.thrust_actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.steer_actions),\
                np.array(self.thrust_actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, steer_action, thrust_action, probs, vals, reward, done):
        self.states.append(state)
        self.steer_actions.append(steer_action)
        self.thrust_actions.append(thrust_action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.steer_actions = []
        self.thrust_actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class SteerActorCriticNetwork(nn.Module):
    def __init__(self,lr):
        super(SteerActorCriticNetwork, self).__init__()
        #layers
        self.LeakyReLU = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.conv1 = nn.Conv2d(1,8,kernel_size = 7, stride = 4,padding = 0)
        self.conv2 = nn.Conv2d(8,16,kernel_size = 3, stride = 1,padding = 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(576+1,512)
        self.fcc2 = nn.Linear(512,1)
        self.fca2 = nn.Linear(512,512)
        self.fca3 = nn.Linear(512,2)
        self.flatten = nn.Flatten()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x,_):
        # reformat image (input = BS,96,96, or 96,96) (output = BS,1,96,96)
        #plot_image(np.squeeze(x))

        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        if(x.dim()==2):
          x = torch.unsqueeze(x,dim=0)
          x = torch.unsqueeze(x,dim=0)
        elif(x.dim()==3):
          x = torch.unsqueeze(x,dim=1)
        subimage = (x[:,:,84:96,13:14]-0.495)*10
        speed = torch.sum(subimage,dim=(2,3))
        x = x[:,:,:84,:]
        
        '''Shared weights'''
        x = self.LeakyReLU(self.conv1(x))
        x = self.pool(x)
        x = self.LeakyReLU(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.cat((x,speed),dim=1)
        x = self.LeakyReLU(self.fc1(x))
        '''Actor and Critic Diverge'''
        #print("actor forward")
        dist = self.LeakyReLU(self.fca2(x))
        dist = self.Softplus(self.fca3(x))
        dist = torch.swapaxes(dist,0,1)
        #print("dist",dist)
        dist = Beta(dist[0],dist[1])
        value = self.fcc2(x)
        #print("value",value)
        
        return dist, value, speed

    def save_checkpoint(self,name):
        torch.save(self.state_dict(),name)

    def load_checkpoint(self,name):
        self.load_state_dict(torch.load(name))


class ThrustActorCriticNetwork(nn.Module):
    def __init__(self,lr):
        super(ThrustActorCriticNetwork, self).__init__()
        #layers
        self.LeakyReLU = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.conv1 = nn.Conv2d(1,8,kernel_size = 7, stride = 4,padding = 0)
        self.conv2 = nn.Conv2d(8,16,kernel_size = 3, stride = 1,padding = 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(576+2,512)
        self.fcc2 = nn.Linear(512,1)
        self.fca2 = nn.Linear(512,512)
        self.fca3 = nn.Linear(512,2)
        self.flatten = nn.Flatten()

        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x,steer_action):
        # reformat image (input = BS,96,96, or 96,96) (output = BS,1,96,96)
        #plot_image(np.squeeze(x))

        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        steer_action = torch.tensor(steer_action).float()
        if(x.dim()==2):
          x = torch.unsqueeze(x,dim=0)
          x = torch.unsqueeze(x,dim=0)
        elif(x.dim()==3):
          x = torch.unsqueeze(x,dim=1)
        subimage = (x[:,:,84:96,13:14]-0.495)*10
        speed = torch.sum(subimage,dim=(2,3))
        x = x[:,:,:84,:]
        
        '''Shared weights'''
        x = self.LeakyReLU(self.conv1(x))
        x = self.pool(x)
        x = self.LeakyReLU(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.cat((x,speed*100),dim=1)
        if(steer_action.dim()==1):
          steer_action = torch.unsqueeze(steer_action,dim=1)
        x = torch.cat((x,steer_action*100),dim=1)
        x = self.LeakyReLU(self.fc1(x))
        '''Actor and Critic Diverge'''
        #print("actor forward")
        dist = self.LeakyReLU(self.fca2(x))
        # Add 1 to prevent convex multimodal shapes
        dist = self.Softplus(self.fca3(x))+1
        dist = torch.swapaxes(dist,0,1)
        #print("dist",dist)
        dist = Beta(dist[0],dist[1])
        value = self.fcc2(x)
        #print("value",value)
        
        return dist, value, speed

    def save_checkpoint(self,name):
        torch.save(self.state_dict(),name)

    def load_checkpoint(self,name):
        self.load_state_dict(torch.load(name))


class Agent:
    def __init__(self, action_space, gamma=0.99, lr=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10,steer_policy="stochastic",thrust_policy = "stochastic"):
        self.action_space=action_space
        if(self.action_space not in ["steer","both"]):
          print("Invalid Action Space")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.action_space = action_space
        self.steer_actor_critic = SteerActorCriticNetwork(lr)
        self.thrust_actor_critic = ThrustActorCriticNetwork(lr)
        self.steer_policy=steer_policy
        self.thrust_policy=thrust_policy
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, steer_action,thrust_action, probs, vals, reward, done):
        self.memory.store_memory(state, steer_action,thrust_action, probs, vals, reward, done)

    def save_models(self,name):
        print('... saving models...')
        if(self.action_space == "steer"):
            self.steer_actor_critic.save_checkpoint(name[0:4]+"steer_"+name[4:])
        elif(self.action_space == "both"):
            self.thrust_actor_critic.save_checkpoint(name[0:4]+"thrust_"+name[4:])

    def load_models(self,steer_name=None,thrust_name=None):
        print('... loading models ...')
        if(steer_name!=None):
            self.steer_actor_critic.load_checkpoint(steer_name)
        if(thrust_name!=None):
            self.thrust_actor_critic.load_checkpoint(thrust_name)

    def choose_action(self, observation):
        self.steer_actor_critic.eval()
        self.thrust_actor_critic.eval()
        state = torch.tensor([observation], dtype=torch.float).to(self.thrust_actor_critic.device)

        '''Get Steering Action'''
        st_dist,st_value,speed = self.steer_actor_critic(state,None)
        if(self.steer_policy == "stochastic"):
          steer_action = st_dist.sample().detach()
        elif(self.steer_policy == "deterministic"):
          steer_action = st_dist.mean.detach()
        else:
          print("Invalid Policy Type")

        '''Get Thrust Action'''
        if(self.action_space == "steer"):
          '''Manually Controlled Thrust'''
          # determine if you are going too fast
          filt = speed<=2.5
          thrust_action = torch.full(speed.shape,0.5)
          thrust_action[filt] = 0.55
        elif(self.action_space == "both"):
          '''Model Controlled Thrust'''
          th_dist,th_value,speed = self.thrust_actor_critic(state,steer_action)
          if(self.thrust_policy == "stochastic"):
            thrust_action = th_dist.sample().detach()
          elif(self.thrust_policy == "deterministic"):
            thrust_action = th_dist.mean.detach()
          else:
            print("Invalid Policy Type")

        #print(thrust_action)

        '''get 𝜋(a|s) for memory'''
        if(self.action_space =="steer"):
          probs = torch.squeeze(torch.exp(st_dist.log_prob(steer_action))).item() 
          action = torch.squeeze(steer_action).item()
        else:
          probs = torch.squeeze(torch.exp(th_dist.log_prob(thrust_action))).item() 
          action = torch.squeeze(thrust_action).item()
        '''get V(s) for memory'''
        if(self.action_space =="steer"):
          value = torch.squeeze(st_value).item()
        else:
          value = torch.squeeze(th_value).item()

        return steer_action, thrust_action, probs, value

    def convert_action(self,state,steer_action,thrust_action):
      # convert action from index, to a list of turning,engine,breaking strengths
      accel = (thrust_action-0.5)*2; accel = max(accel,torch.tensor(0))
      breaking = (0.5-thrust_action)*2; breaking= max(breaking,torch.tensor(0))
      return [steer_action.item()*2 - 1,accel.item(),breaking.item()]


    def learn(self):
        '''choose model to train'''
        if(self.action_space =="steer"):
          model = self.steer_actor_critic
        else:
          model = self.thrust_actor_critic
        model.train()

        for _ in range(self.n_epochs):
            model.train()
            state_arr, steer_action_arr, thrust_action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            steer_action_arr = steer_action_arr.astype('float64')
            thrust_action_arr = thrust_action_arr.astype('float64')
            '''choose actions in training'''
            if(self.action_space == "steer"):
              action_arr = steer_action_arr
            else:
              action_arr = thrust_action_arr

            advantage = np.zeros((len(reward_arr)), dtype=np.float32)

            # Monte Carlo Rewards/ Advantage
            '''
            returns = np.zeros((len(reward_arr)), dtype=np.float32)
            rew = 0
            for t in range(len(reward_arr)-1,-1,-1):
                rew+=reward_arr[t]
                returns[t] = rew
                rew = rew*self.gamma
            _,values,_ = model(state_arr,steer_action_arr)
            values = torch.squeeze(values).detach().numpy()

            advantage = returns - values
            advantage = torch.tensor(advantage).to(model.device)
            '''

            # Advantage Bootsrapped on A = V(s)+r - decay*V(s+1) for various depths
            values = vals_arr
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(model.device)       
            
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(model.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(model.device)
                actions = torch.tensor(action_arr[batch]).to(model.device)

                dist, critic_value,_ = model(states,steer_action_arr[batch])
                critic_value = torch.squeeze(critic_value)

                new_probs = torch.exp(dist.log_prob(actions))
                prob_ratio = new_probs/ old_probs
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                #print("\nActor Loss\n",actor_loss,"\nCritic Loss\n",critic_loss,"\nEntropy\n",dist.entropy().mean())

                total_loss = 2*actor_loss + 0.5*critic_loss
                model.optimizer.zero_grad()
                total_loss.backward()
                model.optimizer.step()

        self.memory.clear_memory() 
