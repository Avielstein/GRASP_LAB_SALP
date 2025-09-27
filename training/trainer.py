"""
Training system for SALP RL experiments.
Handles training loops, evaluation, and model management.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
import torch

from config.base_config import ExperimentConfig
from core.base_agent import BaseAgent, ReplayBuffer, Logger
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


class Trainer:
    """Main trainer class for SALP RL experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Create environment
        self.env = self._create_environment()
        self.eval_env = self._create_environment()  # Separate environment for evaluation
        
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
        
        # Create directories with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config.training.experiment_name}_{timestamp}"
        self.model_dir = os.path.join(config.training.model_dir, self.run_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"Trainer initialized for run: {self.run_name}")
        print(f"Environment: {config.environment.name}")
        print(f"Agent: {config.agent.name}")
        print(f"Device: {self.agent.device}")
        print(f"Models will be saved to: {self.model_dir}")
    
    def _create_environment(self):
        """Create environment based on configuration."""
        env_type = self.config.environment.type
        params = self.config.environment.params
        
        if env_type == "salp_snake":
            return SalpSnakeEnv(
                render_mode=None,  # No rendering during training
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
        """Main training loop."""
        print(f"Starting training for {self.config.training.max_episodes} episodes...")
        
        start_time = time.time()
        
        for episode in range(self.config.training.max_episodes):
            self.episode = episode
            
            # Run episode
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
                
                # Save only the best model (no periodic saves)
                if eval_metrics['mean_score'] > self.best_eval_score:
                    self.best_eval_score = eval_metrics['mean_score']
                    self._save_model("best_model.pth")
                    print(f"New best model saved! Score: {self.best_eval_score:.2f}")
            
            # Print progress
            if episode % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{self.config.training.max_episodes}, "
                      f"Score: {episode_metrics['score']:.2f}, "
                      f"Steps: {episode_metrics['steps']}, "
                      f"Time: {elapsed_time:.1f}s")
        
        # Final save
        self._save_model("final_model.pth")
        self.logger.save_metrics()
        
        print("Training completed!")
        print(f"Best evaluation score: {self.best_eval_score:.2f}")
    
    def _run_episode(self) -> Dict[str, float]:
        """Run a single training episode."""
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
    
    def _evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance."""
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
    
    def load_model(self, filename: str):
        """Load model and training state."""
        filepath = os.path.join(self.model_dir, filename)
        self.agent.load(filepath)
        
        # Load training state if available
        training_filepath = filepath.replace('.pth', '_training_state.pth')
        if os.path.exists(training_filepath):
            training_state = torch.load(training_filepath)
            self.episode = training_state['episode']
            self.total_steps = training_state['total_steps']
            self.best_eval_score = training_state['best_eval_score']
            print(f"Loaded training state from episode {self.episode}")


class EvaluationRunner:
    """Runner for evaluating trained models."""
    
    def __init__(self, config: ExperimentConfig, model_path: str):
        self.config = config
        
        # Create environment with rendering
        env_params = config.environment.params.copy()
        self.env = SalpSnakeEnv(
            render_mode="human",
            width=config.environment.width,
            height=config.environment.height,
            **env_params
        )
        
        # Create and load agent
        self.agent = self._create_agent()
        self.agent.load(model_path)
        
        print(f"Evaluation runner initialized")
        print(f"Model loaded from: {model_path}")
    
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
    
    def run_episodes(self, num_episodes: int = 5, deterministic: bool = True):
        """Run evaluation episodes with rendering."""
        print(f"Running {num_episodes} evaluation episodes...")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            done = False
            truncated = False
            
            while not done and not truncated:
                # Render environment
                self.env.render()
                
                # Select action
                action = self.agent.select_action(obs, deterministic=deterministic)
                
                # Take step
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Small delay for better visualization
                time.sleep(0.01)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Score: {episode_reward:.2f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Food collected: {info.get('food_collected', 0)}")
            print(f"  Collision: {info.get('collision', False)}")
        
        self.env.close()


def main():
    """Demo training script."""
    from config.base_config import get_sac_snake_config
    
    # Get configuration
    config = get_sac_snake_config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
