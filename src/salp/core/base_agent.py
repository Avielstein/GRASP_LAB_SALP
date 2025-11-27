"""
Base classes for RL agents in the SALP project.
Provides common interface and functionality for different agent types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import os
from config.base_config import AgentConfig


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, config: AgentConfig, obs_dim: int, action_dim: int, action_space):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks (to be implemented by subclasses)
        self._build_networks()
    
    @abstractmethod
    def _build_networks(self):
        """Build neural networks for the agent."""
        pass
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update agent parameters using a batch of experiences."""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save agent parameters."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent parameters."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information for logging."""
        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "config": self.config.to_dict()
        }


class BaseNetwork(nn.Module):
    """Base neural network class with common functionality."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        
        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation = activations[activation]()
        self.output_activation = activations[output_activation]() if output_activation else None
        
        # Build network layers
        self.layers = self._build_layers()
    
    def _build_layers(self) -> nn.ModuleList:
        """Build network layers."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_size = self.input_dim
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_dim))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer
        x = self.layers[-1](x)
        
        # Output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0
        self.ptr = 0
        
        # Initialize buffers
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of experiences."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self) -> int:
        return self.size


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {}
        self.episode_metrics = []
    
    def log_scalar(self, key: str, value: float, step: int):
        """Log scalar value."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log episode metrics."""
        metrics['episode'] = episode
        self.episode_metrics.append(metrics)
    
    def save_metrics(self):
        """Save metrics to files."""
        # Save scalar metrics
        for key, values in self.metrics.items():
            filepath = os.path.join(self.log_dir, f"{key}.txt")
            with open(filepath, 'w') as f:
                for step, value in values:
                    f.write(f"{step}\t{value}\n")
        
        # Save episode metrics
        if self.episode_metrics:
            filepath = os.path.join(self.log_dir, "episodes.txt")
            with open(filepath, 'w') as f:
                # Header
                keys = self.episode_metrics[0].keys()
                f.write("\t".join(keys) + "\n")
                
                # Data
                for metrics in self.episode_metrics:
                    values = [str(metrics[key]) for key in keys]
                    f.write("\t".join(values) + "\n")
    
    def get_latest_metric(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1][1]
        return None


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update of target network parameters."""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update of target network parameters."""
    target_net.load_state_dict(source_net.state_dict())
