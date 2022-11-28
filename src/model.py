#Code originally adapted from https://github.com/BY571/Normalized-Advantage-Function-NAF-



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
from collections import deque, namedtuple
import time
import gym
import copy
from torch.distributions import MultivariateNormal, Normal
from torch.nn.functional import sigmoid
import numpy as np
from typing import Optional
import pygame
from pygame import gfxdraw
import gym
from gym import spaces
from gym.utils import seeding
import argparse, sys


class FF(nn.Module):
    #shallow feedforward network
    def __init__(self, state_size, action_size, layer_size, seed):
        super(FF, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.rec = nn.Linear(self.layer_size, layer_size) #not used
        self.action = nn.Linear(layer_size, action_size)
       
        self.x = torch.zeros([layer_size])
        

    
    def forward(self, input):

        self.x = torch.relu(self.head_1(input))
        action = self.action(self.x)
        return action
    
    def reset(self):
        self.x = torch.zeros([layer_size])
    
class LinearPolicy(nn.Module):
    #linear network
    def __init__(self, state_size, action_size, layer_size, seed):
        super(LinearPolicy, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.action = nn.Linear(self.input_shape, action_size)
   
    def forward(self, input):


        action = self.action(input)

        return action

    def reset(self):
        pass
    
    
    

class TwoJoint(gym.Env):

    #Two-joint arm reaching environment
    
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, sparse=False):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.2
        self.t = 0
        self.screen = None
        self.isopen = True
        
        self.MAX_FRAMES = 10
        self.dx_penalty = 0.01
        self.u_penalty = 0.01

        self.screen_dim = 500
        self.resolution = 10 #number of bins for state encoding
        self.sparse = sparse

        high = np.ones([self.resolution*6])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        self.t += 1
        u, v = action.flatten()
        
        u = np.clip(u, -100.0, 100.0)
        v = np.clip(v, -100.0, 100.0)
        
        x, dx, y, dy, goal1, goal2 = self.state  # th := theta

        dt = self.dt


        vecx = np.array([np.cos(x), np.sin(x)])
        vecy = np.array([np.cos(angle_normalize(x+y)), np.sin(angle_normalize(x+y))])
        vec = vecx + vecy
        pos1, pos2 = vec.flatten()
        self.last_u = u
        
        costs = np.sum((pos1-goal1) ** 2 + self.dx_penalty*dx**2 + self.u_penalty*u**2 + (pos2-goal2) ** 2 + self.dx_penalty*dy**2 + self.u_penalty*v**2)
        if self.sparse:
            costs = np.sum((((pos1-goal1) ** 2 + (pos2-goal2) ** 2)>0.5).astype(int) + self.dx_penalty*dx**2 + self.u_penalty*u**2 + self.dx_penalty*dy**2 + self.u_penalty*v**2)

        
        
        raw_distance = np.sum((pos1-goal1) ** 2 + (pos2-goal2) ** 2)

        newdx = dx + u * dt
        newdy = dy + v * dt
        newdx = np.clip(newdx, -2.0, 2.0)
        newdy = np.clip(newdy, -2.0, 2.0)
        newx = x + newdx * dt
        newy = y + newdy * dt
        newx = np.clip(newx, 0, np.pi)
        newy = np.clip(newy, 0, np.pi/2)
        newx = angle_normalize(newx)
        newy = angle_normalize(newy)
        
        self.state = [newx, newdx, newy, newdy, goal1, goal2]
        return self._get_obs(), -costs, self.t >= self.MAX_FRAMES, raw_distance

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        #super().reset()
        self.t = 0
        self.state = [np.random.uniform(low=0, high=np.pi),
                               0, np.random.uniform(low=0, high=np.pi/2),
                               0, 
                      np.random.uniform(low=-1, high=1),
                     1]
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        x, dx, y, dy, goal1, goal2 = self.state
        x = (x-np.pi/2) / (np.pi/2)
        y = (y-np.pi/4) / (np.pi/4)
        dx = dx / 2.0
        dy = dy / 2.0
        resolution = self.resolution
        step = 2.0 / resolution
        statex = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if x >= ii-1e-8 and x <= ii + step+1e-8:
                statex[idx] = 1.0
                
        statedx = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if dx >= ii-1e-8 and dx <= ii + step+1e-8:
                statedx[idx] = 1.0
                

        statey = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if y >= ii-1e-8 and y <= ii + step+1e-8:
                statey[idx] = 1.0
                
        statedy = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if dy >= ii-1e-8 and dy <= ii + step+1e-8:
                statedy[idx] = 1.0
                
                
        stateg1 = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if goal1 >= ii-1e-8 and goal1 <= ii + step+1e-8:
                stateg1[idx] = 1.0
                
        stateg2 = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if goal2 >= ii-1e-8 and goal2 <= ii + step+1e-8:
                stateg2[idx] = 1.0
                
                
                
        state = np.concatenate([statex, statey, statedx, statedy, stateg1, stateg2], 0)
        return state


class Maze(gym.Env):


    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, sparse=False):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.2
        self.t = 0
        self.screen = None
        self.isopen = True
        
        self.MAX_FRAMES = 10
        self.dx_penalty = 0.01
        self.u_penalty = 0.01

        self.screen_dim = 500
        self.resolution = 10 #number of bins for state encoding
        self.sparse = sparse

        high = np.ones([self.resolution*6])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        self.t += 1
        u, v = action.flatten()
        
        u = np.clip(u, -100.0, 100.0)
        v = np.clip(v, -100.0, 100.0)
        
        x, dx, y, dy, goal1, goal2 = self.state

        dt = self.dt


        pos1, pos2 = x, y

        self.last_u = u 
        costs = np.sum((pos1-goal1) ** 2 + self.dx_penalty*dx**2 + self.u_penalty*u**2 + (pos2-goal2) ** 2 + self.dx_penalty*dy**2 + self.u_penalty*v**2)

        if self.sparse:
            costs = np.sum((((pos1-goal1) ** 2 + (pos2-goal2) ** 2)>0.5).astype(int) + self.dx_penalty*dx**2 + self.u_penalty*u**2 + self.dx_penalty*dy**2 + self.u_penalty*v**2)

                              
        raw_distance = np.sum((pos1-goal1) ** 2 + (pos2-goal2) ** 2)

        newdx = dx + u * dt
        newdy = dy + v * dt
        newdx = np.clip(newdx, -2.0, 2.0)
        newdy = np.clip(newdy, -2.0, 2.0)
        newx = x + newdx * dt
        newy = y + newdy * dt
        newx = np.clip(newx, -1, 1)
        newy = np.clip(newy, -1, 1)
        newx = angle_normalize(newx)
        newy = angle_normalize(newy)
        
        self.state = [newx, newdx, newy, newdy, goal1, goal2]
        return self._get_obs(), -costs, self.t >= self.MAX_FRAMES, raw_distance

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        #super().reset()
        self.t = 0
        self.state = [np.random.uniform(low=-1, high=1),
                               np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1),
                               np.random.uniform(low=-1, high=1), 
                      np.random.uniform(low=-1, high=1),
                     np.random.uniform(low=-1, high=1)]
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        x, dx, y, dy, goal1, goal2 = self.state
        
        dx = dx / 2.0
        dy = dy / 2.0
        resolution = self.resolution
        step = 2.0 / resolution
        statex = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if x >= ii-1e-8 and x <= ii + step+1e-8:
                statex[idx] = 1.0
                
        statedx = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if dx >= ii-1e-8 and dx <= ii + step+1e-8:
                statedx[idx] = 1.0
                

        statey = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if y >= ii-1e-8 and y <= ii + step+1e-8:
                statey[idx] = 1.0
                
        statedy = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if dy >= ii-1e-8 and dy <= ii + step+1e-8:
                statedy[idx] = 1.0
                
                
        stateg1 = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if goal1 >= ii-1e-8 and goal1 <= ii + step+1e-8:
                stateg1[idx] = 1.0
                
        stateg2 = np.zeros([resolution])
        idx = -1
        for ii in np.arange(-1.0, 1.0-1e-8, 2.0/resolution):
            idx += 1
            if goal2 >= ii-1e-8 and goal2 <= ii + step+1e-8:
                stateg2[idx] = 1.0
                
                
                
        state = np.concatenate([statex, statey, statedx, statedy, stateg1, stateg2], 0)
        return state


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class NAF(nn.Module):
    #NAF refers to Normalized Advantage Functions (Gu et al., 2016)
    def __init__(self, state_size, action_size, layer_size, seed, num_layers=2, controller_freq=0.0, explore_amount=1.0,
                A_coef = 1.0, learn_A_coef=False, combo_type="sample"):
        super(NAF, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape, layer_size) #shared first layer
        self.action_values = nn.Linear(layer_size, action_size) #actor output
        self.value = nn.Linear(layer_size, 1) #critic outpt
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2)) #not used
        self.sigma = nn.Linear(layer_size, self.action_size)
        
        self.explore_amount = explore_amount
        
        self.controller_freq = controller_freq
        
        self.learn_controller_freq = nn.Linear(self.input_shape, 1) #used if combo_type = learnfunction or learnconstant
        
        self.combo_type = combo_type
        
        
        if learn_A_coef:
            self.A_coef = torch.nn.Parameter(torch.Tensor([A_coef]))
        else:
            self.A_coef = A_coef
        

    
    def forward(self, input, action=None, explore=2):

        x = input
        x = torch.relu(self.head_1(x))

        action_value = self.action_values(x)
        
        V = self.value(x) #critic output
        
        action_value = action_value.unsqueeze(-1)
        
        P = torch.eye(self.action_size).unsqueeze(0).to(device)*(torch.exp(self.sigma(x))**2).unsqueeze(-1) #not used
        Q = A = None
        if action is not None:
            A = -self.A_coef*torch.sum((action.flatten()-action_value.flatten())**2) #action surprise

            Q = A + V
        

        noise = 0
        #explore = 0: use deterministic output of RL module
        #explore = 1: combine deterministic output of RL module and output of external controller
        #explore = 2: combine output of RL module (with exploration noise) and output of external controller (with noise)
        if explore == 1 or explore == 2:
            if explore == 2:
                dist = Normal(action_value.squeeze(-1), self.explore_amount)
                action = dist.sample()
                
                dist_noise = Normal(0, self.explore_amount)
                noise = dist_noise.sample()
            if explore == 1:
                action = action_value.squeeze(-1)
            
            orig_action = action #orig_action refers to output of RL module
            if self.combo_type == "sample":
                if np.random.rand() < self.controller_freq:
                    which = 1 #indicates that external controller is in charge
                    action = (self.fixed_policy(input.squeeze(-1).squeeze(-1))+noise)#torch.rand(1).to(device) * 2 - 1
                else:
                    which = 0 #indicates that RL module is in charge
             
            if self.combo_type == "mean":
                control = self.controller_freq
                action = (1-control) * action + control*(self.fixed_policy(input.squeeze(-1).squeeze(-1)) + noise)
                which = 0
                
            if self.combo_type == "learnfunction":
                control = sigmoid(self.learn_controller_freq(input))
                action = (1-control) * action + control*(self.fixed_policy(input.squeeze(-1).squeeze(-1)) + noise)
                which = 0
                
            if self.combo_type == "learnconstant":
                control = sigmoid(self.learn_controller_freq(input*0))
                action = (1-control) * action + control*(self.fixed_policy(input.squeeze(-1).squeeze(-1)) + noise)
                which = 0
                
            if self.combo_type == "sum":
                control = self.controller_freq
                action = action + (self.fixed_policy(input.squeeze(-1).squeeze(-1)) + noise)
                which = 0

        if explore == 0:
            action = action_value.squeeze(-1)
            orig_action = action
            which = 0
            

        action = action.detach()
        
        return action, orig_action, A, V, which
    
    
    
class DQN_Agent():

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 NUPDATES,
                 device,
                 seed, controller_freq=0.0, use_Q=True, explore_amount=1.0, A_coef=1.0, 
                 learn_A_coef=False, combo_type="sample"):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.NUPDATES = NUPDATES
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        
        self.use_Q = use_Q


        self.action_step = 4
        self.last_action = None

        self.qnetwork_local = NAF(state_size, action_size,layer_size, seed, 
                                  controller_freq=controller_freq, explore_amount=explore_amount, A_coef=A_coef,
                                 learn_A_coef=learn_A_coef, combo_type=combo_type).to(device)


        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=LR)
                
        self.t_step = 0
        
    
    def step(self, state, action, orig_action, reward, next_state, done, explore=2):#, writer):

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            Q_losses = []
            for _ in range(self.NUPDATES):
                experience_type = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
                experiences = (state, action, orig_action, reward, next_state, done)
                if explore==2:
                    loss = self.learn(experiences)
                    self.Q_updates += 1
                    Q_losses.append(loss)

    def act(self, state, explore=2):

        self.qnetwork_local.eval()
        with torch.no_grad():
            action, orig_action, _, V, which = self.qnetwork_local(state.unsqueeze(0), explore=explore)
            _, _, A_, V_, which = self.qnetwork_local(state.unsqueeze(0), action=action, explore=explore)
            Q = A_ + V_

        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy(), orig_action.cpu().squeeze().numpy(), Q.cpu().squeeze().numpy(), V.cpu().squeeze().numpy()

    def learn(self, experiences):
        self.optimizer.zero_grad()
        states, actions, orig_actions, rewards, next_states, dones = experiences
        actions = torch.from_numpy(np.array([[actions]])).float().to(self.device)
        orig_actions = torch.from_numpy(np.array([[orig_actions]])).float().to(self.device)
        
        with torch.no_grad():
            _, _, _, V_, which = self.qnetwork_local(next_states)

        V_targets = rewards + (self.GAMMA * V_ * (1 - dones))
        
        _, _, A, V, which = self.qnetwork_local(states, actions)
        
        _, _, A_orig, _, _ = self.qnetwork_local(states, orig_actions)

        #print('ADVANTAGE', A)
        Q = A + V
        #print('heyy', A, V)
        # Compute loss
        if self.use_Q == "yes":
            #loss = F.mse_loss(Q, V_targets)
            actor_loss =  F.mse_loss(V.detach() + A, V_targets.detach()) #gradient with respect to A is proportional to RPE (including action surprise term) times (a-mu)
            critic_loss = F.mse_loss(V + A.detach(), V_targets.detach())#gradient with respect to V is proportional to RPE (including action surprise term)
            loss = actor_loss + critic_loss
        elif self.use_Q == "no":
            actor_loss = F.mse_loss(V.detach()+A, V_targets.detach()+A.detach()) #gradient with respect to A is proportional to RPE times (a-mu)
            critic_loss = F.mse_loss(V, V_targets.detach()) #gradient with respect to V is proportional to RPE
            loss = actor_loss + critic_loss
        elif self.use_Q == "only_when_in_control": #experimental setting: use action surprise only when external controller is in charge of action (only well-defined when combo_type == "sample")
            actor_loss = F.mse_loss(V.detach()+A, V_targets.detach()+A.detach()) #gradient with respect to A is proportional to RPE (including action surprise term) times (a-mu)
            critic_loss = F.mse_loss(V, V_targets.detach()) #gradient with respect to V is proportional to RPE (including action surprise term)
            loss = actor_loss + critic_loss
            if which == 1:
                loss = loss * 0
                
        elif self.use_Q == "no_without_efferent":
            actor_loss = F.mse_loss(V.detach()+A_orig, V_targets.detach()+A_orig.detach()) #gradient with respect to A is proportional to RPE (including action surprise term) times (a-mu)
            critic_loss = F.mse_loss(V, V_targets.detach()) #gradient with respect to V is proportional to RPE (including action surprise term)
            loss = actor_loss + critic_loss
                
        elif self.use_Q.startswith("sup"): #like "yes" but with additional imitation (supervision) loss
            sup_level = float(self.use_Q[4:]) #format example: "sup_1.0" where 1.0 is the coefficient of the auxiliary imitation (supervision) loss
            actor_loss = F.mse_loss(V.detach() + A, V_targets.detach()-sup_level*A.detach())
            critic_loss = F.mse_loss(V + A.detach(), V_targets.detach())
            loss = actor_loss + critic_loss
        


        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()


        
        return loss.detach().cpu().numpy()            


    
def run(frames=1000, fixed_policy=None, test_freq=2):

        scores = [] 
        scores_combo = []
        scores_explore = []
        frame = 0
        i_episode = 0
        state = env.reset()
        state = torch.from_numpy(state).float().to(device)
        score = 0 
        actions = []
        velocities = []
        positions = []
        dcoords = []
        coords = []
        goals = []
        Qs = []
        Vs = []
        rewards = []
        last_vec = np.nan
        raw_distances = []
        for frame in range(1, frames+1):
            agent.qnetwork_local.fixed_policy = fixed_policy
            for param in agent.qnetwork_local.fixed_policy.parameters():
                param.requires_grad = False

            #explore = 0: use deterministic output of RL module ("scores")
            #explore = 1: combine deterministic output of RL module and output of external controller ("scores_combo")
            #explore = 2: combine output of RL module (with exploration noise) and output of external controller (with noise) ("scores_explore")

            explore = i_episode % (test_freq+2)
            if explore >= 2:
                explore = 2
                
                
            action, orig_action, Q, V = agent.act(state, explore=explore)
            actions.append(action)
            Qs.append(Q)
            Vs.append(V)

            x, dx, y, dy, g1, g2 = env.state
            
            if task == "TwoJoint":
                vecx = np.array([np.cos(x), np.sin(x)])
                vecy = np.array([np.cos(angle_normalize(x+y)), np.sin(angle_normalize(x+y))])
                vec = vecx + vecy
            elif task == "Maze":
                vec = np.array([x, y])
            
            velocities.append(vec-last_vec)
            positions.append(vec)
            
            goals.append(np.array([g1, g2]))
            
            last_vec = vec
            
            coords.append(np.array([x, y]))
            dcoords.append(np.array([dx, dy]))
            
            next_state, reward, done, raw_distance = env.step(np.array([action]))
                              
            

            next_state = torch.from_numpy(next_state).float().to(device)
            agent.step(state, action, orig_action, reward, next_state, done, explore=explore)#, writer)


            rewards.append(reward)
            raw_distances.append(raw_distance)

            state = next_state
            score += reward

            if done:

                if explore == 0:

                    scores.append(score)              

                if explore == 1:
                    scores_combo.append(score)
                if explore == 2:
                    scores_explore.append(score)
                    
                i_episode +=1 
                state = env.reset()
                state = torch.from_numpy(state).float().to(device)
                score = 0  
                last_vec = np.nan

        return scores, scores_combo, scores_explore, actions, Qs, Vs, rewards, raw_distances, velocities, positions, dcoords, coords, goals
    
test_freq = 2
seed = 0
BUFFER_SIZE = 1
BATCH_SIZE = 1
GAMMA = 0.99
TAU = 1e-3

UPDATE_EVERY = 1
NUPDATES = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using ", device)

scores_list = []


parser=argparse.ArgumentParser()

parser.add_argument('--trial', type=int) #trial index
parser.add_argument('--explore_amount', type=float) #exploration noise
parser.add_argument('--fp_name', type=str) #name of fixed policy (external controller) (ff_trained -- expert, ff_trained_short -- intermediate, ff_trained_random -- random network policy)
parser.add_argument('--controller_freq', type=float) #if combo_type == "sample": likelihood that action is driven by external controller.  If combo_type == "mean": weight of external controller contribution to action.
parser.add_argument('--LR', type=float) #learning rate
parser.add_argument('--use_Q', type=str) #use Q-learning (action surprise) instead of vanilla ator-critic
parser.add_argument('--num_frames', type=int)
parser.add_argument('--A_coef', type=float) #for action surprise model this controls the value of 1/sigma.  For vanilla actor critic controls relative learning rate of actor compared to critic
parser.add_argument('--learn_A_coef', type=int) #whether A_coef is fixed or trained
parser.add_argument('--task', type=str) #Maze or TwoJoint
parser.add_argument('--combo_type', type=str) 
#sample, mean, learnconstant, learnfunction, sum.
#sample: action drawn from RL output with some probability, else mixture.  
#mean: action = convex combination of the contributions of RL output and external controller output. 
#learnconstant: like "mean", but combination weights are a learned constant. 
#learnfunction: like "mean," but combination weights are a learned function of state.
#sum: action = sum of the contributions of RL output and external controller output. 
parser.add_argument('--sparse', type=int, default=0) #sparse rewards or graded rewards (default is graded)
parser.add_argument('--output_directory', type=str, default="/home/jwl2182/ActionDopamineDistributedControl/output") #sparse rewards or graded rewards (default is graded)




args = parser.parse_args()

trial = args.trial
explore_amount = args.explore_amount
fp_name = args.fp_name
controller_freq = args.controller_freq
LR = args.LR
use_Q = args.use_Q
num_frames = args.num_frames
A_coef = args.A_coef
learn_A_coef = bool(args.learn_A_coef)
combo_type = args.combo_type
task = args.task
sparse=args.sparse
OUTPUT_DIRECTORY = args.output_directory

seed = trial
np.random.seed(seed)
target = np.random.rand()*2-1

if task == "Maze":
    env = Maze(sparse=sparse)
    fixed_policy_dict = {}
    fp1 = FF(60, 2, 256, 0).to(device)
    fp1.load_state_dict(torch.load("../output/ff_maze_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained"] = fp1

    fp1a = FF(60, 2, 256, 0).to(device)
    fp1a.load_state_dict(torch.load("../output/ff_maze_short_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained_short"] = fp1a

    fp1c = FF(60, 2, 256, 0).to(device)
    fp1c.load_state_dict(torch.load("../output/ff_maze_random_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained_random"] = fp1c

    '''
    fp3 = LinearPolicy(60, 2, 256, 0).to(device)
    fp3.load_state_dict(torch.load("linear_maze_constant1.pth"))
    fixed_policy_dict["linear_constant1"] = fp3
    '''
    
elif task == "TwoJoint":
    env = TwoJoint(sparse=sparse)
    fixed_policy_dict = {}
    fp1 = FF(60, 2, 256, 0).to(device)
    fp1.load_state_dict(torch.load("../output/ff_two_joint_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained"] = fp1

    fp1a = FF(60, 2, 256, 0).to(device)
    fp1a.load_state_dict(torch.load("../output/ff_two_joint_short_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained_short"] = fp1a

    
    fp1c = FF(60, 2, 256, 0).to(device)
    fp1c.load_state_dict(torch.load("../output/ff_two_joint_random_"+str(trial)+".pth"))
    fixed_policy_dict["ff_trained_random"] = fp1c

    '''
    fp3 = LinearPolicy(60, 2, 256, 0).to(device)
    fp3.load_state_dict(torch.load("linear_two_joint_constant1.pth"))
    fixed_policy_dict["linear_constant1"] = fp3
    '''

else:
    print('Task', task)
    assert False
    
fixed_policy = fixed_policy_dict[fp_name]

env.seed(seed)
action_size     = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
agent = DQN_Agent(state_size=state_size,    
                    action_size=action_size,
                    layer_size=5000,
                    BATCH_SIZE=BATCH_SIZE, 
                    BUFFER_SIZE=BUFFER_SIZE, 
                    LR=LR, 
                    TAU=TAU, 
                    GAMMA=GAMMA, 
                    UPDATE_EVERY=UPDATE_EVERY,
                    NUPDATES=NUPDATES,
                    device=device, 
                    seed=seed, controller_freq=controller_freq, use_Q = use_Q,
                 explore_amount=explore_amount, A_coef=A_coef, learn_A_coef=learn_A_coef, combo_type=combo_type)

agent.qnetwork_local.action_values.bias.data = torch.Tensor(np.array([0.0])).to(device)
agent.qnetwork_local.action_values.weight.data *= 0
agent.qnetwork_local.value.weight.data *= 0
agent.qnetwork_local.action_values.bias.requires_grad = False

for param in agent.qnetwork_local.head_1.parameters():
    param.requires_grad = False

cur_scores, cur_scores_combo, cur_scores_explore, actions, Qs, Vs, rewards, raw_distances, velocities, positions, dcoords, coords, goals = run(frames = num_frames,
                                                              fixed_policy=fixed_policy)

np.save(OUTPUT_DIRECTORY+"/"+"v3_cur_scores_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), cur_scores) #scores of RL policy alone

np.save(OUTPUT_DIRECTORY+"/"+"v3_cur_scores_combo_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), cur_scores_combo) #scores of combined policy

np.save(OUTPUT_DIRECTORY+"/"+"v3_cur_scores_explore_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), cur_scores_explore) #scores of combined policy including exploration noise


np.save(OUTPUT_DIRECTORY+"/"+"v3_raw_distances_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), raw_distances)


np.save(OUTPUT_DIRECTORY+"/"+"v3_Qs_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), Qs)

np.save(OUTPUT_DIRECTORY+"/"+"v3_Vs_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), Vs)

np.save(OUTPUT_DIRECTORY+"/"+"v3_rewards_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), rewards)

np.save(OUTPUT_DIRECTORY+"/"+"v3_actions_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), actions)

np.save(OUTPUT_DIRECTORY+"/"+"v3_velocities_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), velocities)

np.save(OUTPUT_DIRECTORY+"/"+"v3_positions_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), positions)

np.save(OUTPUT_DIRECTORY+"/"+"v3_coords_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), coords)

np.save(OUTPUT_DIRECTORY+"/"+"v3_dcoords_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), dcoords)

np.save(OUTPUT_DIRECTORY+"/"+"v3_goals_trial_"+str(trial)+"_explore_amount_"+str(explore_amount)+"_fp_name_"+str(fp_name)+"_controller_freq_"+str(controller_freq)+"_LR_"+str(LR)+"_use_Q_"+str(use_Q)+"_A_coef_"+str(A_coef)+"_learnA_"+str(learn_A_coef)+"_task_"+str(task)+"_combo_"+str(combo_type)+"_sparse_"+str(sparse), goals)