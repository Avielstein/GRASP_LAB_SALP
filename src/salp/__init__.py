"""
SALP - Salp-inspired Approach to Low-energy Propulsion
Bio-inspired soft underwater robot simulation for reinforcement learning research.
"""

__version__ = "0.1.0"

from salp.agents import (
    SACAgent,
    SACGAILAgent,
    SB3SACAgent,
    Discriminator,
)
from salp.environments import SalpSnakeEnv
from salp.training import Trainer, ContinuousTrainer, ExpertBuffer
from salp.core import BaseAgent
from salp.config import BaseConfig

__all__ = [
    "SACAgent",
    "SACGAILAgent", 
    "SB3SACAgent",
    "Discriminator",
    "SalpSnakeEnv",
    "Trainer",
    "ContinuousTrainer",
    "ExpertBuffer",
    "BaseAgent",
    "BaseConfig",
]
