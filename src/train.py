import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import os
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.97, 
          'buffer_size': 10000,
          'epsilon_min': 0.02,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000, 
          'epsilon_delay_decay': 100, 
          'batch_size': 750, 
          'gradient_steps': 5,
          'update_target_strategy': 'replace',
          'update_target_freq': 500, 
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()}

class ProjectAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.myDQN(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        buffer_size=config['buffer_size']
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model,observation)

    def save(self):
        self.path ="trained_model.pkl"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() +"/src/trained_model.pkl"
        self.model = self.myDQN(device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return 
    
    def greedy_action(self, myDQN, state):
        device = "cuda" if next(myDQN.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = myDQN(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
    
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    #I implement a more robust DQN version than the one in class
    def myDQN(self, device):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons=256

        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
            ).to(device)

        return DQN

    def train(self):
        print('Device:', self.device)
        max_episode = 300 
        best_value = 0
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        ## TRAIN NETWORK
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select action
            action=self.act(state,np.random.rand()< epsilon)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()

            # update target network if needed
            if self.update_target_strategy == 'replace': 
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            
            step += 1
            if done or trunc:
                episode += 1
                
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:.3e}'.format(episode_cum_reward),
                      ", evaluation score ", '{:.3e}'.format(validation_score),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0

                if validation_score > best_value:
                    best_value = validation_score
                    self.best_model = deepcopy(self.model).to(self.device)
                    self.save()
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        self.save()
        return episode_return

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity 
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)