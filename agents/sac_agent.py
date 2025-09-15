"""
Soft Actor-Critic (SAC) agent implementation for SALP environments.
Based on the SAC algorithm with entropy regularization and automatic temperature tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple
import os

from core.base_agent import BaseAgent, BaseNetwork, soft_update
from config.base_config import AgentConfig


class GaussianPolicy(BaseNetwork):
    """Gaussian policy network for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list, 
                 activation: str = "relu", action_scale: float = 1.0):
        # Output dimension is 2 * action_dim (mean and log_std)
        super().__init__(obs_dim, 2 * action_dim, hidden_sizes, activation)
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Constrain log_std to reasonable range
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        x = super().forward(obs)
        
        # Split output into mean and log_std
        mean, log_std = torch.chunk(x, 2, dim=-1)
        
        # Constrain log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t) * self.action_scale
            
            # Calculate log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - action.pow(2) / (self.action_scale ** 2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class QNetwork(BaseNetwork):
    """Q-value network for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list, activation: str = "relu"):
        super().__init__(obs_dim + action_dim, 1, hidden_sizes, activation)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with observation and action concatenated."""
        x = torch.cat([obs, action], dim=-1)
        return super().forward(x)


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent implementation."""
    
    def __init__(self, config: AgentConfig, obs_dim: int, action_dim: int, action_space):
        self.action_scale = float(action_space.high[0])  # Assume symmetric action space
        super().__init__(config, obs_dim, action_dim, action_space)
        
        # SAC-specific parameters
        self.alpha = config.params.get('alpha', 0.2)
        self.target_entropy = config.params.get('target_entropy', -action_dim)
        self.alpha_lr = config.params.get('alpha_lr', config.learning_rate)
        
        # Automatic temperature tuning
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        
        # Initialize target networks
        soft_update(self.q1_target, self.q1, 1.0)
        soft_update(self.q2_target, self.q2, 1.0)
    
    def _build_networks(self):
        """Build SAC networks."""
        # Policy network
        self.policy = GaussianPolicy(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_sizes,
            self.config.activation,
            self.action_scale
        ).to(self.device)
        
        # Q-networks
        self.q1 = QNetwork(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_sizes,
            self.config.activation
        ).to(self.device)
        
        self.q2 = QNetwork(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_sizes,
            self.config.activation
        ).to(self.device)
        
        # Target Q-networks
        self.q1_target = QNetwork(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_sizes,
            self.config.activation
        ).to(self.device)
        
        self.q2_target = QNetwork(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_sizes,
            self.config.activation
        ).to(self.device)
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(obs_tensor, deterministic)
            return action.cpu().numpy()[0]
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update SAC agent using a batch of experiences."""
        # Convert batch to tensors
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
        
        # Update Q-networks
        q_loss, q1_values, q2_values = self._update_q_networks(obs, actions, rewards, next_obs, dones)
        
        # Update policy
        policy_loss, log_probs = self._update_policy(obs)
        
        # Update temperature (alpha)
        alpha_loss = self._update_temperature(log_probs)
        
        # Update target networks
        soft_update(self.q1_target, self.q1, self.config.tau)
        soft_update(self.q2_target, self.q2, self.config.tau)
        
        self.training_step += 1
        
        return {
            'q_loss': q_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha,
            'q1_mean': q1_values.mean().item(),
            'q2_mean': q2_values.mean().item()
        }
    
    def _update_q_networks(self, obs: torch.Tensor, actions: torch.Tensor, 
                          rewards: torch.Tensor, next_obs: torch.Tensor, 
                          dones: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Update Q-networks."""
        # Current Q-values
        q1_values = self.q1(obs, actions)
        q2_values = self.q2(obs, actions)
        
        # Target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_obs)
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # Q-network losses
        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        return q_loss.item(), q1_values, q2_values
    
    def _update_policy(self, obs: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Update policy network."""
        # Sample actions from current policy
        actions, log_probs = self.policy.sample(obs)
        
        # Q-values for sampled actions
        q1_values = self.q1(obs, actions)
        q2_values = self.q2(obs, actions)
        q_values = torch.min(q1_values, q2_values)
        
        # Policy loss (maximize Q-value - entropy penalty)
        policy_loss = (self.alpha * log_probs - q_values).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), log_probs
    
    def _update_temperature(self, log_probs: torch.Tensor) -> float:
        """Update temperature parameter (alpha)."""
        # Temperature loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Update temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update alpha value
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def save(self, filepath: str):
        """Save agent parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config.to_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
