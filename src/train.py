import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import os
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

# Configuration de l'environnement
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

# Configuration des hyperparamètres
config = {
    'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.97,
    'buffer_size': 10000,
    'epsilon_min': 0.02,
    'epsilon_max': 1.0,
    'epsilon_decay_period': 20000,
    'epsilon_delay_decay': 100,
    'batch_size': 750,
    'gradient_steps': 5,
    'update_target_strategy': 'replace',
    'update_target_freq': 500,
    'update_target_tau': 0.005,
    'criterion': nn.SmoothL1Loss()
}

class CustomAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay_steps = config['epsilon_decay_period']
        self.epsilon_delay = config.get('epsilon_delay_decay', 20)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.criterion = config.get('criterion', nn.MSELoss())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        self.gradient_steps = config.get('gradient_steps', 1)
        self.update_strategy = config.get('update_target_strategy', 'replace')
        self.target_update_freq = config.get('update_target_freq', 20)
        self.tau = config.get('update_target_tau', 0.005)

    def choose_action(self, observation, explore=False):
        if explore:
            return env.action_space.sample()
        return self._predict_action(self.model, observation)

    def save_model(self, filepath="hiv_agent_model.pkl"):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath="hiv_agent_model.pkl"):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def _predict_action(self, model, state):
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
            return torch.argmax(q_values).item()

    def _perform_gradient_step(self):
        if len(self.memory) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            q_targets = rewards + self.gamma * (1 - dones) * self.target_model(next_states).max(1)[0].detach()
            q_values = self.model(states).gather(1, actions.to(torch.long).unsqueeze(1))
            loss = self.criterion(q_values, q_targets.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _create_model(self, device):
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        hidden_units = 256

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        ).to(device)

        return model

    def train(self):
        max_episodes = 300
        best_eval_score = float('-inf')
        episode_rewards = []
        epsilon = self.epsilon_max
        state, _ = env.reset()
        step_counter = 0

        for episode in range(max_episodes):
            cumulative_reward = 0
            done = False

            while not done:
                if step_counter > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

                action = self.choose_action(state, explore=(random.random() < epsilon))
                next_state, reward, done, truncated, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done or truncated)
                cumulative_reward += reward
                state = next_state
                step_counter += 1

                for _ in range(self.gradient_steps):
                    self._perform_gradient_step()

                if self.update_strategy == 'replace' and step_counter % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                elif self.update_strategy == 'ema':
                    for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if done or truncated:
                    break

            eval_score = evaluate_HIV(self, 1)
            episode_rewards.append(cumulative_reward)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                self.save_model()

            state, _ = env.reset()
            print(f"Episode: {episode}, Reward: {cumulative_reward}, Eval Score: {eval_score}")

        return episode_rewards

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)