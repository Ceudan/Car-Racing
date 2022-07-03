from torch._C import dtype
import gym
from gym.wrappers.monitoring import video_recorder
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.record_video import RecordVideo
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay

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
import pickle
import statistics as st
from gym.core import RewardWrapper
import gc

import os  

from utils import*
from models import*


def official_test(steer_model=None, thrust_model = None, steer_policy = "stochastic",thrust_policy = "stochastic", early_stop = True,num_tests = 100):
  env = wrap_env(gym.make("CarRacing-v1").unwrapped,render = False,early_stop=early_stop)
  action_space = "steer"
  if(thrust_model!=None):
    action_space = "both"
  agent = Agent(action_space=action_space,steer_policy=steer_policy,thrust_policy=thrust_policy)
  agent.load_models(steer_name= steer_model,thrust_name=thrust_model)

  test_rews = []
  test_ep_lens = []
  for i in range(0,num_tests):
    rew,ep_len = simulate(agent=agent,env=env,steer_model=steer_model, thrust_model=thrust_model, steer_policy=steer_policy,thrust_policy=thrust_policy)
    test_rews.append(rew)
    test_ep_lens.append(ep_len)
    print("Test",i)
    print("rew =",rew,"ep_len =",ep_len)
    print("av rew =",st.mean(test_rews),"ep_len =",st.mean(test_ep_lens),"\n")
  return test_rews,test_ep_lens


def simulate(agent=None,env=None,steer_model=None, thrust_model = None, steer_policy = "stochastic",thrust_policy = "stochastic",render = False):
    if(env==None):
       env = wrap_env(gym.make("CarRacing-v1").unwrapped,render = render)
    if(agent==None):
      action_space = "steer"
      if(thrust_model!=None):
        action_space = "both"
      agent = Agent(action_space=action_space,steer_policy=steer_policy,thrust_policy=thrust_policy)
      agent.load_models(steer_name= steer_model,thrust_name=thrust_model)
    observation = env.reset()
    done = False
    score = 0
    #skip first few useless episodes
    state, reward, done, info = env.skip_episodes(70,[0,0.5,0])
    while not done:
        steer_action, thrust_action, prob, val = agent.choose_action(state)
        next_state, reward, done, info = env.step(agent.convert_action(state,steer_action,thrust_action))
        state = next_state
    print("score",env.real_rew,"ep_len",env.ep_len)
    env.env.close()
    if(render):
        show_video()
    return env.real_rew,env.ep_len


def plot_learning_curve(scores):
    x = [i+1 for i in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 10 scores')
    plt.show()

def show_video():       
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")   
  
def get_speed(state):
    subimage = (state[84:96,13:14]-0.495)*10
    speed = np.sum(subimage)
    return speed

def process_image(image):
  # process image
  red = image[:,:,0:1]*0.55
  green = image[:,:,1:2]*-0.45 + 255*0.495
  image = np.squeeze(red+green)/255
  image[image<0.4] = 0

  return image

def plot_image(array):
  plt.imshow(array, cmap='gray', vmin=0, vmax=1)
  plt.show()

def load_list(filename):
  with open(filename, 'rb') as filehandle:
    l = pickle.load(filehandle)
  return l

def save_list(l,filename):
  with open(filename, 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(l, filehandle)
       
    
class wrap_env:
 def __init__(self,env,render=False,timestep = 5, init_bad_steps = 0, bad_step_limit = 20,early_stop=True):
   self.env = env
   self.env.reset()
   self.timestep = 5
   self.render = False
   self.bad_steps = init_bad_steps
   self.bad_step_limit = bad_step_limit
   self.ep_len=0
   self.real_rew = 0
   self.early_stop=early_stop
   if(render):
     self.env = RecordVideo(self.env, './video')
     self.env.render()
   
 def reset(self):
   self.ep_len = 0
   self.real_rew = 0
   return self.env.reset()
 def skip_episodes(self,num_episodes,action_code):
   for i in range(0,num_episodes):
     state, rew, done, info = self.env.step(action_code)
     self.real_rew+=rew
     self.ep_len+=1
     if(done):
       break
   return process_image(state), rew, done, info
 def step(self,action_code):
   step_rew = 0
   for i in range(0,self.timestep):
     state, rew, done, info = self.env.step(action_code)
     self.real_rew+=rew
     self.ep_len+=1
     # keep track of all rewards in timestep
     step_rew+=rew
     #step_rew=0
     # limit number of bad steps
     if(rew<0):
       self.bad_steps+=1
     else:
       self.bad_steps=0
     if(self.bad_steps>=self.bad_step_limit and self.early_stop):
       self.bad_steps = 0
       done = True
       #step_rew+=-100
     if(done):
       break
   return process_image(state), step_rew, done, info