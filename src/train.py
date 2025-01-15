from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
from collections import deque
from copy import deepcopy
import tqdm

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# Baseline Q-network
class QNetwork_baseline(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_baseline, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Model 1: Dueling Q-network
class QNetwork_dueling(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_dueling, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Value stream
        self.fc3_value = nn.Linear(128, 1)
        
        # Advantage stream
        self.fc3_advantage = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = self.fc3_value(x)
        advantage = self.fc3_advantage(x)
        
        return value + advantage - advantage.mean()
    
# Model 2: Finetuned Q-network
class QNetwork_finetuned(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_finetuned, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_size, 256), 
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2))
        self.fc3 = nn.Sequential(nn.Linear(128, 64), 
                                 nn.ReLU())
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)
    
# Model 3: Q-network with Attention

class QNetwork_attention(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4):
        """
        A Double DQL model with attention.

        Args:
        - state_dim: The dimensionality of the state space.
        - action_dim: The number of possible actions.
        - embed_dim: The embedding size used for attention.
        - num_heads: The number of attention heads.
        - hidden_dim: The size of the hidden layers after attention.
        """
        super(QNetwork_attention, self).__init__()

        # State embedding layer
        self.embedding = nn.Linear(state_dim, 128)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(128, num_heads, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
        - state: A batch of states (batch_size, state_dim).

        Returns:
        - Q-values for all actions (batch_size, action_dim).
        """
        # Embed the state
        x = self.embedding(state)  # (batch_size, state_dim) -> (batch_size, embed_dim=128)

        # Add sequence dimension for attention (treat state_dim as sequence length)
        x = x.unsqueeze(1)  # (batch_size, embed_dim=128) -> (batch_size, seq_len=1, embed_dim=128)

        # Attention mechanism (no explicit queries/keys, self-attention is used)
        attn_output, _ = self.attention(x, x, x)  # (batch_size, 1, embed_dim=128)

        # Remove sequence dimension
        attn_output = attn_output.squeeze(1)  # (batch_size, embed_dim=128)

        # Fully connected layers for Q-values
        x = F.relu(self.fc1(attn_output))
        q_values = self.fc2(x)  # (batch_size, action_dim)

        return q_values
    
# Model 4: Fine-tuned Dueling Q-network

class QNetwork_finetuned_dueling(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_finetuned_dueling, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        
        # Value stream
        self.fc3_value = nn.Linear(128, 1)
        
        # Advantage stream
        self.fc3_advantage = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.norm1(self.fc1(x)))
        x = torch.relu(self.norm2(self.fc2(x)))
        
        value = self.fc3_value(x)
        advantage = self.fc3_advantage(x)
        
        return value + advantage - advantage.mean()
    
# Model 5: Better finetuned Q-network

class QNetwork_better(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_better, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_size, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256,256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, action_size))
        
    def forward(self, x):
        return self.fc(x)
                                 
# Model 6: Better dueling Q-network

class QNetwork_better_dueling(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork_better_dueling, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_size, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256,256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU())
        
        # Value stream
        self.fc_value = nn.Linear(256, 1)
        
        # Advantage stream
        self.fc_advantage = nn.Linear(256, action_size)
        
    def forward(self, x):
        x = self.fc(x)
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        
        return value + advantage - advantage.mean()


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    # Sample experiences from the replay buffer
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert lists of arrays into single NumPy arrays for optimized conversion into PyTorch tensors
        states = np.array([e[0] if isinstance(e[0], np.ndarray) else e[0][0] for e in experiences if e is not None and (isinstance(e[0], np.ndarray) or (isinstance(e[0], tuple) and isinstance(e[0][0], np.ndarray)))], dtype=np.float32)
        actions = np.array([e[1] for e in experiences if e is not None], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences if e is not None], dtype=np.float32)
        next_states = np.array([e[3] for e in experiences if e is not None], dtype=np.float32)
        dones = np.array([e[4] for e in experiences if e is not None], dtype=np.float32)

        # Convert NumPy arrays to PyTorch tensors
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class ProjectAgent:
    config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_dueling(state_size, action_size),
            'criterion': nn.SmoothL1Loss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 200,
            'tau': 5e-4
        }
    
    def __init__(self):
        self.state_dim = state_size
        self.nb_actions = action_size
        self.gamma = self.config['gamma'] if 'gamma' in self.config.keys() else 0.95
        
        # Initialize the replay buffer
        self.batch_size = self.config['batch_size'] if 'batch_size' in self.config.keys() else 100
        self.buffer_size = self.config['buffer_size'] if 'buffer_size' in self.config.keys() else int(1e5)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Initialize the epsilon-greedy parameters
        self.epsilon_start = self.config['epsilon_start'] if 'epsilon_start' in self.config.keys() else 1.0
        self.epsilon_end = self.config['epsilon_end'] if 'epsilon_end' in self.config.keys() else 0.01
        self.epsilon_decay = self.config['epsilon_decay'] if 'epsilon_decay' in self.config.keys() else 0.995
        self.epsilon = self.epsilon_start
        
        # Initialize the Q-networks
        self.qnetwork_local = self.config['qnetwork_local'] if 'qnetwork_local' in self.config.keys() else QNetwork_baseline(state_size, action_size)
        self.qnetwork_target = deepcopy(self.qnetwork_local)
        self.criterion = self.config['criterion'] if 'criterion' in self.config.keys() else torch.nn.MSELoss()
        
        # Initialize the optimizer
        lr = self.config['learning_rate'] if 'learning_rate' in self.config.keys() else 0.001
        self.optimizer = self.config['optimizer'] if 'optimizer' in self.config.keys() else torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.set_scheduler = self.config['set_scheduler'] if 'set_scheduler' in self.config.keys() else False
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.nb_gradient_steps = self.config['gradient_steps'] if 'gradient_steps' in self.config.keys() else 1
        
        # Initialize the target network update strategy
        self.update_target_strategy = self.config['update_target_strategy'] if 'update_target_strategy' in self.config.keys() else 'replace'
        self.update_target_freq = self.config['update_target_freq'] if 'update_target_freq' in self.config.keys() else 20
        self.tau = self.config['tau'] if 'tau' in self.config.keys() else 1e-3
        self.step = 0
        
    def act(self, observation, use_random=False):
        if use_random or random.random() < self.epsilon:
            return random.randint(0, self.nb_actions - 1)
        else:
            with torch.no_grad():
                
            # Check if observation is an array with a dictionary, and extract only the array part
                if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[0], np.ndarray):
                    observation = observation[0]  # Extract the numpy array
                    
                norm_observation = np.sign(observation) * np.log(1 + np.abs(observation))    # Normalize the observation
                state = torch.tensor(norm_observation, dtype=torch.float32).unsqueeze(0)
                q_values = self.qnetwork_local(state)
                return int(torch.argmax(q_values).item())
            
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        (states, actions, rewards, next_states, dones) = self.memory.sample()
        
        # Compute the local Q-values
        actions = actions.unsqueeze(1)
        q_values = self.qnetwork_local(states).gather(1, actions).squeeze(1)
        
        # Compute the target Q-values
        with torch.no_grad():
            next_q_values = self.qnetwork_local(next_states)
            max_actions = torch.argmax(next_q_values, dim=1).unsqueeze(1)
            next_target_q_values = self.qnetwork_target(next_states).gather(1, max_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_target_q_values * (1 - dones)
            
        # Loss and optimization
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.set_scheduler:
            self.scheduler.step()
        
        # Update the target network
        self.step += 1
        if self.update_target_strategy == 'replace' and self.step % self.update_target_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        elif self.update_target_strategy == 'soft' and self.step % self.update_target_freq == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def save(self, path="agent_checkpoint.pth"):
        torch.save({
            'q_network': self.qnetwork_local.state_dict(),
            'target_network': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self):
        checkpoint = torch.load('src/Saved_models/agent_checkpoint_dueling.pth')
        self.qnetwork_local.load_state_dict(checkpoint['q_network'])
        self.qnetwork_target.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
            
    
# Training Pipeline
    
# Initialize the agent
    
""" config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_better_dueling(state_size, action_size),
            'criterion': nn.MSELoss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'tau': 1e-3
        }

agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
#plt.plot(scores)
#plt.xlabel('Episode')
#plt.ylabel('Score')
#plt.show()

# Save the agent
agent.save("agent_checkpoint_dueling3.pth")



config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_better_dueling(state_size, action_size),
            'criterion': nn.SmoothL1Loss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'tau': 1e-3
        }

agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
#plt.plot(scores)
#plt.xlabel('Episode')
#plt.ylabel('Score')
#plt.show()

# Save the agent
agent.save("agent_checkpoint_dueling3L1Loss.pth")

config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_better_dueling(state_size, action_size),
            'criterion': nn.MSELoss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'soft',
            'update_target_freq': 20,
            'tau': 5e-4
        }

agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
#plt.plot(scores)
#plt.xlabel('Episode')
#plt.ylabel('Score')
#plt.show()

# Save the agent
agent.save("agent_checkpoint_dueling3_soft.pth")


config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_better_dueling(state_size, action_size),
            'criterion': nn.MSELoss(),
            'set_scheduler': True,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'tau': 1e-3
        }

agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
#plt.plot(scores)
#plt.xlabel('Episode')
#plt.ylabel('Score')
#plt.show()

# Save the agent
agent.save("agent_checkpoint_dueling3_scheduler.pth")


config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_better(state_size, action_size),
            'criterion': nn.MSELoss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'tau': 1e-3
        }

agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.add((state, action, reward, next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
#plt.plot(scores)
#plt.xlabel('Episode')
#plt.ylabel('Score')
#plt.show()

# Save the agent
agent.save("agent_checkpoint_better.pth")
 """
""" 
config = {  'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'qnetwork_local': QNetwork_dueling(state_size, action_size),
            'criterion': nn.SmoothL1Loss(),
            'set_scheduler': False,
            'learning_rate': 0.001,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 20,
            'tau': 1e-3
        }
        
        
agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state, _ = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        norm_state = np.sign(state) * np.log(1 + np.abs(state))    # Normalize the state
        norm_next_state = np.sign(next_state) * np.log(1 + np.abs(next_state))    # Normalize the next state    
        agent.memory.add((norm_state, action, reward, norm_next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        
        
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()

# Save the agent
agent.save("agent_checkpoint_dueling_normalization.pth") """


agent = ProjectAgent()

# Training loop
n_episodes = 1000
scores = []
scores_window = deque(maxlen=100)

for episode in tqdm.tqdm(range(n_episodes)):
    state, _ = env.reset()
    score = 0
    cpt_buffer = 0
    max_steps = 200
    step = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        norm_state = np.sign(state) * np.log(1 + np.abs(state))    # Normalize the state
        norm_next_state = np.sign(next_state) * np.log(1 + np.abs(next_state))    # Normalize the next state    
        agent.memory.add((norm_state, action, reward, norm_next_state, done))
        agent.learn()
        state = next_state
        score += reward
        step += 1
        
        if done or step >= max_steps:
            done = True
            break
            
    scores.append(score)
    scores_window.append(score)
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Average Score: {np.mean(scores_window)}')
        

# Save the agent
agent.save("agent_checkpoint_dueling_normalization.pth")