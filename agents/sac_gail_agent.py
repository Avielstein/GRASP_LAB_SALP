"""
SAC agent enhanced with GAIL (Generative Adversarial Imitation Learning).
Combines Soft Actor-Critic with discriminator-based reward shaping.
"""

import torch
import numpy as np
from typing import Dict, Optional
import os

from agents.sac_agent import SACAgent
from agents.discriminator import Discriminator
from config.base_config import AgentConfig
from training.expert_buffer import ExpertBuffer


class SACGAILAgent(SACAgent):
    """
    SAC agent with GAIL integration.
    
    Uses a discriminator to provide reward signal based on expert demonstrations.
    Supports hybrid reward (environment + GAIL) or pure imitation.
    """
    
    def __init__(self, config: AgentConfig, obs_dim: int, action_dim: int, action_space,
                 expert_buffer: Optional[ExpertBuffer] = None,
                 use_gail: bool = True,
                 reward_env_weight: float = 0.3,
                 reward_gail_weight: float = 0.7,
                 discriminator_update_freq: int = 1):
        """
        Initialize SAC+GAIL agent.
        
        Args:
            config: Agent configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_space: Action space
            expert_buffer: Buffer containing expert demonstrations
            use_gail: Whether to use GAIL rewards
            reward_env_weight: Weight for environment reward
            reward_gail_weight: Weight for GAIL reward
            discriminator_update_freq: Update discriminator every N policy updates
        """
        # Initialize parent SAC agent
        super().__init__(config, obs_dim, action_dim, action_space)
        
        # GAIL parameters
        self.use_gail = use_gail
        self.reward_env_weight = reward_env_weight
        self.reward_gail_weight = reward_gail_weight
        self.initial_env_weight = reward_env_weight
        self.initial_gail_weight = reward_gail_weight
        self.discriminator_update_freq = discriminator_update_freq
        self.expert_buffer = expert_buffer
        
        # Performance-based adaptive weights
        self.use_adaptive_weights = False
        self.performance_history = []
        self.performance_window = 20  # Rolling average over N episodes
        self.performance_thresholds = {
            'beginner': 300,    # < 300: Need heavy guidance
            'learning': 600,    # 300-600: Reducing guidance
            'competent': 800,   # 600-800: Allow exploration
            'expert': 1000      # 800-1000: Optimize freely
                               # > 1000: Full autonomy
        }
        self.weight_schedule = {
            'beginner': (0.30, 0.70),   # (env, gail)
            'learning': (0.50, 0.50),
            'competent': (0.70, 0.30),
            'expert': (0.85, 0.15),
            'master': (0.95, 0.05)
        }
        
        # Initialize discriminator if GAIL is enabled
        if self.use_gail:
            discriminator_lr = config.params.get('discriminator_lr', config.learning_rate)
            self.discriminator = Discriminator(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=config.hidden_sizes,
                activation=config.activation,
                learning_rate=discriminator_lr,
                device=self.device
            )
            
            if expert_buffer is None:
                print("Warning: GAIL enabled but no expert buffer provided. "
                      "GAIL rewards will not be available until expert buffer is set.")
        else:
            self.discriminator = None
    
    def set_expert_buffer(self, expert_buffer: ExpertBuffer):
        """Set or update the expert buffer."""
        self.expert_buffer = expert_buffer
        print(f"Expert buffer set: {expert_buffer}")
    
    def compute_gail_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Compute GAIL reward for a single transition.
        
        Args:
            obs: Observation
            action: Action taken
        
        Returns:
            GAIL reward value
        """
        if not self.use_gail or self.discriminator is None:
            return 0.0
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        reward = self.discriminator.predict_reward(obs_tensor, action_tensor)
        return reward.item()
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update SAC agent with GAIL enhancements.
        
        Args:
            batch: Batch of agent experiences
        
        Returns:
            Dictionary with training metrics
        """
        # Standard SAC update
        sac_metrics = super().update(batch)
        
        # GAIL discriminator update
        if self.use_gail and self.discriminator is not None and self.expert_buffer is not None:
            # Update discriminator at specified frequency
            if self.training_step % self.discriminator_update_freq == 0:
                try:
                    # Sample expert batch
                    expert_batch = self.expert_buffer.sample(self.config.batch_size)
                    
                    # Update discriminator
                    disc_metrics = self.discriminator.update(expert_batch, batch)
                    
                    # Merge metrics
                    sac_metrics.update(disc_metrics)
                except ValueError as e:
                    # Expert buffer might be empty
                    print(f"Warning: Could not update discriminator: {e}")
        
        return sac_metrics
    
    def compute_hybrid_reward(self, env_reward: float, obs: np.ndarray, 
                             action: np.ndarray) -> float:
        """
        Compute hybrid reward combining environment and GAIL rewards.
        
        Args:
            env_reward: Reward from environment
            obs: Current observation
            action: Action taken
        
        Returns:
            Combined reward
        """
        if not self.use_gail or self.discriminator is None:
            return env_reward
        
        # Compute GAIL reward
        gail_reward = self.compute_gail_reward(obs, action)
        
        # Weighted combination
        hybrid_reward = (self.reward_env_weight * env_reward + 
                        self.reward_gail_weight * gail_reward)
        
        return hybrid_reward
    
    def save(self, filepath: str):
        """Save agent parameters including discriminator."""
        # Save base SAC agent
        super().save(filepath)
        
        # Save discriminator if it exists
        if self.discriminator is not None:
            disc_filepath = filepath.replace('.pth', '_discriminator.pth')
            self.discriminator.save(disc_filepath)
            print(f"Discriminator saved to {disc_filepath}")
    
    def load(self, filepath: str):
        """Load agent parameters including discriminator."""
        # Load base SAC agent
        super().load(filepath)
        
        # Load discriminator if it exists
        if self.discriminator is not None:
            disc_filepath = filepath.replace('.pth', '_discriminator.pth')
            if os.path.exists(disc_filepath):
                self.discriminator.load(disc_filepath)
                print(f"Discriminator loaded from {disc_filepath}")
            else:
                print(f"Warning: Discriminator file not found: {disc_filepath}")
    
    def get_info(self) -> Dict:
        """Get agent information including GAIL status."""
        info = super().get_info()
        
        gail_info = {
            'use_gail': self.use_gail,
            'reward_env_weight': self.reward_env_weight,
            'reward_gail_weight': self.reward_gail_weight,
            'discriminator_update_freq': self.discriminator_update_freq
        }
        
        if self.expert_buffer is not None:
            gail_info['expert_buffer_size'] = len(self.expert_buffer)
            gail_info['expert_episodes'] = len(self.expert_buffer.episodes)
        
        info['gail'] = gail_info
        return info
    
    def set_reward_weights(self, env_weight: float, gail_weight: float):
        """
        Dynamically adjust reward weights during training.
        
        Useful for curriculum learning: start with high GAIL weight,
        gradually increase environment weight.
        
        Args:
            env_weight: Weight for environment reward
            gail_weight: Weight for GAIL reward
        """
        self.reward_env_weight = env_weight
        self.reward_gail_weight = gail_weight
        print(f"Reward weights updated: env={env_weight:.2f}, gail={gail_weight:.2f}")
    
    def get_discriminator_accuracy(self, agent_batch: Dict[str, np.ndarray]) -> float:
        """
        Get current discriminator accuracy on agent vs expert.
        
        Args:
            agent_batch: Batch of agent experiences
        
        Returns:
            Discriminator accuracy (0 to 1)
        """
        if not self.use_gail or self.discriminator is None or self.expert_buffer is None:
            return 0.0
        
        try:
            expert_batch = self.expert_buffer.sample(len(agent_batch['observations']))
            
            # Convert to tensors
            expert_obs = torch.FloatTensor(expert_batch['observations']).to(self.device)
            expert_actions = torch.FloatTensor(expert_batch['actions']).to(self.device)
            agent_obs = torch.FloatTensor(agent_batch['observations']).to(self.device)
            agent_actions = torch.FloatTensor(agent_batch['actions']).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                expert_probs = self.discriminator.forward(expert_obs, expert_actions)
                agent_probs = self.discriminator.forward(agent_obs, agent_actions)
                
                expert_correct = (expert_probs > 0.5).float().mean()
                agent_correct = (agent_probs <= 0.5).float().mean()
                accuracy = (expert_correct + agent_correct) / 2.0
            
            return accuracy.item()
        except:
            return 0.0
    
    def enable_adaptive_weights(self, enable: bool = True):
        """Enable or disable performance-based adaptive weight adjustment."""
        self.use_adaptive_weights = enable
        if enable:
            print("🎓 Adaptive GAIL weights ENABLED - agent will earn independence through performance")
        else:
            print("🎓 Adaptive GAIL weights DISABLED - using fixed weights")
    
    def update_performance(self, episode_score: float):
        """
        Update performance history and adjust weights if adaptive mode enabled.
        
        Args:
            episode_score: Score from completed episode
        """
        self.performance_history.append(episode_score)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # Update weights if adaptive mode is enabled
        if self.use_adaptive_weights and len(self.performance_history) >= self.performance_window:
            self._update_adaptive_weights()
    
    def _get_competence_level(self) -> str:
        """
        Determine current competence level based on recent performance.
        
        Returns:
            Competence level: 'beginner', 'learning', 'competent', 'expert', or 'master'
        """
        if len(self.performance_history) < self.performance_window:
            return 'beginner'
        
        avg_score = np.mean(self.performance_history)
        
        if avg_score < self.performance_thresholds['beginner']:
            return 'beginner'
        elif avg_score < self.performance_thresholds['learning']:
            return 'learning'
        elif avg_score < self.performance_thresholds['competent']:
            return 'competent'
        elif avg_score < self.performance_thresholds['expert']:
            return 'expert'
        else:
            return 'master'
    
    def _update_adaptive_weights(self):
        """Update reward weights based on current performance level."""
        old_env_weight = self.reward_env_weight
        old_gail_weight = self.reward_gail_weight
        
        level = self._get_competence_level()
        new_env_weight, new_gail_weight = self.weight_schedule[level]
        
        # Only update if weights changed significantly
        if abs(new_env_weight - old_env_weight) > 0.01:
            self.reward_env_weight = new_env_weight
            self.reward_gail_weight = new_gail_weight
            
            avg_score = np.mean(self.performance_history)
            print(f"\n🎓 PERFORMANCE MILESTONE REACHED!")
            print(f"   Level: {level.upper()}")
            print(f"   Avg Score (last {self.performance_window}): {avg_score:.1f}")
            print(f"   Weights: Env {new_env_weight:.0%} | GAIL {new_gail_weight:.0%}")
            print(f"   Agent earning more independence! 🚀\n")
    
    def get_adaptive_status(self) -> Dict:
        """Get current adaptive weight status."""
        if not self.use_adaptive_weights:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'level': self._get_competence_level() if len(self.performance_history) >= self.performance_window else 'warming_up',
            'avg_score': np.mean(self.performance_history) if self.performance_history else 0,
            'episodes_tracked': len(self.performance_history),
            'env_weight': self.reward_env_weight,
            'gail_weight': self.reward_gail_weight
        }
