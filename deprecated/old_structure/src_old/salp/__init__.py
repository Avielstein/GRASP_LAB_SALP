"""
SALP - Salp-inspired Approach to Low-energy Propulsion
Bio-inspired soft underwater robot simulation for reinforcement learning research.

Uses well-tested libraries (Stable Baselines3) for RL algorithms.
Custom implementations have been removed - use SB3SACAgent for training.
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
# Import what you need directly from submodules instead

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

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "SalpSnakeEnv":
        from salp.environments.salp_snake_env import SalpSnakeEnv
        return SalpSnakeEnv
    elif name == "SalpRobotEnv":
        from salp.environments.salp_robot_env import SalpRobotEnv
        return SalpRobotEnv
    elif name == "SB3SACAgent":
        from salp.agents.sb3_sac_agent import SB3SACAgent
        return SB3SACAgent
    elif name == "Discriminator":
        from salp.agents.discriminator import Discriminator
        return Discriminator
    elif name in ["Trainer", "ContinuousTrainer", "ExpertBuffer"]:
        # These have import issues, skip for now
        raise ImportError(f"{name} has dependency issues, import directly")
    elif name in ["BaseNetwork", "soft_update", "hard_update"]:
        from salp.core import base_agent
        return getattr(base_agent, name, None)
    elif name in ["AgentConfig", "ExperimentConfig", "get_sac_snake_config"]:
        from salp.config import base_config
        return getattr(base_config, name, None)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
