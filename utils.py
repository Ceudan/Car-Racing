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
import numpy as np
from torch import nn
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
 def __init__(self,env,render=False,timestep = 5, init_bad_steps = 0, bad_step_limit = 20):
   self.env = env
   self.env.reset()
   self.timestep = 5
   self.render = False
   self.bad_steps = init_bad_steps
   self.bad_step_limit = bad_step_limit
   self.ep_len=0
   self.real_rew = 0
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
     if(self.bad_steps>=self.bad_step_limit):
       self.bad_steps = 0
       done = True
       step_rew+=-100
     if(done):
       break
   return process_image(state), step_rew, done, info   
