"""
SALP - Salp-inspired Approach to Low-energy Propulsion
Bio-inspired soft underwater robot simulation for reinforcement learning research.

Uses well-tested libraries (Stable Baselines3) for RL algorithms.
Custom implementations have been removed - use SB3SACAgent for training.
"""

__version__ = "0.1.0"

from salp.agents import SB3SACAgent, Discriminator
from salp.environments import SalpRobotEnv, SalpSnakeEnv
from salp.training import Trainer, ContinuousTrainer, ExpertBuffer
from salp.core import BaseNetwork, soft_update, hard_update
from salp.config import AgentConfig, ExperimentConfig, get_sac_snake_config

__all__ = [
    "SB3SACAgent",
    "Discriminator",
    "SalpRobotEnv",
    "SalpSnakeEnv",
    "Trainer",
    "ContinuousTrainer",
    "ExpertBuffer",
    "BaseNetwork",
    "soft_update",
    "hard_update",
    "AgentConfig",
    "ExperimentConfig",
    "get_sac_snake_config",
]
