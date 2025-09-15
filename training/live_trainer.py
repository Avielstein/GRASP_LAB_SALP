"""
Live training system with visual feedback for SALP RL experiments.
Shows the agent learning in real-time with periodic visual episodes.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
import torch
import threading
import queue

from config.base_config import ExperimentConfig
from core.base_agent import BaseAgent, ReplayBuffer, Logger
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


class LiveTrainer:
    """Trainer with live visual feedback during training."""
    
    def __init__(self, config: ExperimentConfig, visual_frequency: int = 10):
        self.config = config
        self.visual_frequency = visual_frequency  # Show visual episode every N episodes
        
        # Create training environment (no rendering)
        self.env = self._create_environment(render_mode=None)
        self.eval_env = self._create_environment(render_mode=None)
        
        # Create visual environment (with rendering)
        self.visual_env = self._create_environment(render_mode="human")
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.agent.buffer_size,
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )
        
        # Create logger
        log_dir = os.path.join(config.training.log_dir, config.training.experiment_name)
        self.logger = Logger(log_dir)
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = -float('inf')
        
        # Create directories
        self.model_dir = os.path.join(config.training.model_dir, config.training.experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"Live Trainer initialized for experiment: {config.training.experiment_name}")
        print(f"Environment: {config.environment.name}")
        print(f"Agent: {config.agent.name}")
        print(f"Device: {self.agent.device}")
        print(f"Visual episodes every {visual_frequency} episodes")
    
    def _create_environment(self, render_mode=None):
        """Create environment based on configuration."""
        env_type = self.config.environment.type
        params = self.config.environment.params
        
        if env_type == "salp_snake":
            return SalpSnakeEnv(
                render_mode=render_mode,
                width=self.config.environment.width,
                height=self.config.environment.height,
                **params
            )
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _create_agent(self):
        """Create agent based on configuration."""
        agent_type = self.config.agent.type
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        if agent_type == "sac":
            return SACAgent(
                config=self.config.agent,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_space=self.env.action_space
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train(self):
        """Main training loop with live visualization."""
        print(f"Starting live training for {self.config.training.max_episodes} episodes...")
        print("Visual episodes will show the agent's current performance!")
        
        start_time = time.time()
        
        for episode in range(self.config.training.max_episodes):
            self.episode = episode
            
            # Decide if this should be a visual episode
            is_visual_episode = (episode % self.visual_frequency == 0 and episode > 0)
            
            if is_visual_episode:
                print(f"\n🎬 VISUAL EPISODE {episode} - Watch the agent perform!")
                episode_metrics = self._run_visual_episode()
                print(f"Visual episode completed: Score: {episode_metrics['score']:.2f}, "
                      f"Food: {episode_metrics['food_collected']}")
            else:
                # Run normal training episode
                episode_metrics = self._run_episode()
            
            # Log episode metrics
            self.logger.log_episode(episode, episode_metrics)
            
            # Update agent episode count
            self.agent.episode_count = episode
            
            # Evaluation
            if episode % self.config.training.eval_frequency == 0:
                eval_metrics = self._evaluate()
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    self.logger.log_scalar(f"eval_{key}", value, episode)
                
                # Save best model
                if eval_metrics['mean_score'] > self.best_eval_score:
                    self.best_eval_score = eval_metrics['mean_score']
                    self._save_model("best_model.pth")
                    print(f"🏆 New best model saved! Score: {self.best_eval_score:.2f}")
            
            # Save model periodically
            if episode % self.config.training.save_frequency == 0:
                self._save_model(f"model_episode_{episode}.pth")
            
            # Print progress
            if episode % 25 == 0 or is_visual_episode:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{self.config.training.max_episodes}, "
                      f"Score: {episode_metrics['score']:.2f}, "
                      f"Food: {episode_metrics['food_collected']}, "
                      f"Steps: {episode_metrics['steps']}, "
                      f"Time: {elapsed_time:.1f}s")
        
        # Final save
        self._save_model("final_model.pth")
        self.logger.save_metrics()
        
        print("🎉 Training completed!")
        print(f"Best evaluation score: {self.best_eval_score:.2f}")
        
        # Close visual environment
        self.visual_env.close()
    
    def _run_episode(self) -> Dict[str, float]:
        """Run a single training episode (no rendering)."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        
        done = False
        truncated = False
        
        while not done and not truncated and episode_steps < self.config.training.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(obs, deterministic=False)
            
            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Store experience
            self.replay_buffer.add(obs, action, reward, next_obs, done or truncated)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Train agent
            if (len(self.replay_buffer) >= self.config.training.start_training_after and 
                self.total_steps % self.config.training.train_frequency == 0):
                
                batch = self.replay_buffer.sample(self.config.agent.batch_size)
                loss_info = self.agent.update(batch)
                episode_losses.append(loss_info)
                
                # Log training metrics
                for key, value in loss_info.items():
                    self.logger.log_scalar(f"train_{key}", value, self.total_steps)
            
            obs = next_obs
        
        # Calculate episode metrics
        episode_metrics = {
            'score': episode_reward,
            'steps': episode_steps,
            'food_collected': info.get('food_collected', 0),
            'collision': info.get('collision', False)
        }
        
        # Add loss metrics if available
        if episode_losses:
            avg_losses = {}
            for key in episode_losses[0].keys():
                avg_losses[f'avg_{key}'] = np.mean([loss[key] for loss in episode_losses])
            episode_metrics.update(avg_losses)
        
        return episode_metrics
    
    def _run_visual_episode(self) -> Dict[str, float]:
        """Run a visual episode showing the agent's current performance."""
        obs, _ = self.visual_env.reset()
        episode_reward = 0
        episode_steps = 0
        
        done = False
        truncated = False
        
        print("🎮 Visual episode starting - watch the SALP learn!")
        
        while not done and not truncated and episode_steps < self.config.training.max_steps_per_episode:
            # Render environment
            self.visual_env.render()
            
            # Select action (deterministic for cleaner visualization)
            action = self.agent.select_action(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = self.visual_env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Small delay for better visualization
            time.sleep(0.02)  # 50 FPS
        
        # Calculate episode metrics
        episode_metrics = {
            'score': episode_reward,
            'steps': episode_steps,
            'food_collected': info.get('food_collected', 0),
            'collision': info.get('collision', False),
            'visual_episode': True
        }
        
        return episode_metrics
    
    def _evaluate(self, num_episodes: int = 3) -> Dict[str, float]:
        """Evaluate agent performance (reduced episodes for faster training)."""
        scores = []
        food_collected_list = []
        steps_list = []
        
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            
            done = False
            truncated = False
            
            while not done and not truncated and episode_steps < self.config.training.max_steps_per_episode:
                # Select action deterministically
                action = self.agent.select_action(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_steps += 1
            
            scores.append(episode_reward)
            food_collected_list.append(info.get('food_collected', 0))
            steps_list.append(episode_steps)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_food_collected': np.mean(food_collected_list),
            'mean_steps': np.mean(steps_list)
        }
    
    def _save_model(self, filename: str):
        """Save model and training state."""
        filepath = os.path.join(self.model_dir, filename)
        self.agent.save(filepath)
        
        # Save training state
        training_state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_eval_score': self.best_eval_score,
            'config': self.config
        }
        
        training_filepath = filepath.replace('.pth', '_training_state.pth')
        torch.save(training_state, training_filepath)


def main():
    """Demo live training script."""
    from config.base_config import get_sac_snake_config
    
    # Get configuration
    config = get_sac_snake_config()
    config.training.max_episodes = 200  # Shorter for demo
    
    # Create live trainer
    trainer = LiveTrainer(config, visual_frequency=10)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
