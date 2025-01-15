import os
from typing import *

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from memory_storage import PrioritizedReplayBuffer, ReplayBuffer
from agent_network import Network

# Environnement avec limite de temps
hiv_environment = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
EPISODE_LIMIT = 200  # Nombre maximum de steps par épisode


# Agent principal
class AdaptiveAgent:
    def __init__(self):
        self.agent_core = RLTrainer()

    def choose_action(self, state, random=False):
        return self.agent_core.decide_action(state, random)

    def save_model(self, save_path):
        print(f"Saving model at {save_path}")
        self.agent_core.save_checkpoint(save_path)

    def load_model(self, load_path):
        print(f"Loading model from {load_path}")
        self.agent_core.load_checkpoint(load_path)


class RLTrainer:
    def __init__(
        self,
        reward_multiplier: float = 1e7,
        buffer_capacity: int = int(5e5),
        mini_batch_size: int = 1024,
        stacked_frames: int = 4,
        learning_rate: float = 1e-4,
        regularization_coeff: float = 1e-3,
        gradient_clipping: float = 500.0,
        target_net_sync: int = 2500,
        max_exploration: float = 0.9,
        min_exploration: float = 0.1,
        exploration_decay_rate: float = 1 / 300,
        gamma: float = 0.98,
        training_steps: int = 2,
        enable_domain_randomization: bool = False,
        hidden_layer_size: int = 512,
        enable_per: bool = True,
        alpha: float = 0.5,
        beta: float = 0.4,
        beta_growth_rate: float = 0.00001,
        priority_epsilon: float = 1e-5,
        use_double_dqn: bool = True,
    ):
        # Initialisation des paramètres d'entraînement
        self.reward_multiplier = reward_multiplier
        self.enable_domain_randomization = enable_domain_randomization
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=self.enable_domain_randomization),
            max_episode_steps=EPISODE_LIMIT,
        )
        self._configure_randomization(self.enable_domain_randomization)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Hyperparamètres
        self.mini_batch_size = mini_batch_size
        self.gradient_clipping = gradient_clipping
        self.target_net_sync = target_net_sync
        self.epsilon = max_exploration
        self.max_exploration = max_exploration
        self.min_exploration = min_exploration
        self.exploration_decay_rate = exploration_decay_rate
        self.gamma = gamma
        self.training_steps = training_steps
        self.stacked_frames = stacked_frames
        self.device = torch.device("cpu")

        # Initialisation du buffer de mémoire
        self.enable_per = enable_per
        if enable_per:
            self.priority_epsilon = priority_epsilon
            self.memory = PrioritizedReplayBuffer(
                obs_dim=obs_dim,
                size=buffer_capacity,
                batch_size=mini_batch_size,
                alpha=alpha,
                beta=beta,
                beta_increment_per_sampling=beta_growth_rate,
            )
        else:
            self.memory = ReplayBuffer(
                obs_dim=obs_dim,
                size=buffer_capacity,
                batch_size=mini_batch_size,
            )

        # Réseaux DQN
        self.use_double_dqn = use_double_dqn
        dqn_config = dict(
            in_dim=(stacked_frames + 1) * obs_dim,
            nf=hidden_layer_size,
            out_dim=action_dim,
        )
        self.q_network = Network(**dqn_config).to(self.device)
        self.target_network = Network(**dqn_config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=regularization_coeff)

        # États internes
        self.training_mode = False
        self.episode_start = 1
        self.current_step = 0
        self.state_buffer = None

    def decide_action(self, observation, random=False):
        """Décider d'une action à partir d'un état."""
        processed_state = np.log10(observation + 1e-5)
        if self.current_step == 0:
            self.state_buffer = np.stack([processed_state] * (self.stacked_frames + 1), axis=0)
        else:
            self.state_buffer[: self.stacked_frames, :] = self.state_buffer[1:]
            self.state_buffer[-1, :] = processed_state
        self.current_step = (self.current_step + 1) % EPISODE_LIMIT
        return self._select_action(self.state_buffer.reshape(-1))

    def _select_action(self, state: np.ndarray) -> int:
        """Implémentation d'une stratégie epsilon-greedy."""
        state = state.reshape((-1, 6))[:, np.array([0, 2, 1, 3, 4, 5])].reshape(-1)
        if self.epsilon > np.random.random() and not self.training_mode:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.q_network(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def _configure_randomization(self, enable: bool) -> None:
        """Configurer la randomisation du domaine."""
        self.env.unwrapped.domain_randomization = enable

    def save_checkpoint(self, save_path: str) -> None:
        """Sauvegarder les poids du modèle."""
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print("Checkpoint sauvegardé !")

    def load_checkpoint(self, load_path: str) -> None:
        """Charger les poids sauvegardés."""
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        print("Checkpoint chargé avec succès !")