"""
Continuous visual training system for SALP RL experiments.
Shows the agent running constantly with real-time updates when new best models are found.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
import torch
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from config.base_config import ExperimentConfig
from core.base_agent import BaseAgent, ReplayBuffer, Logger
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


class ContinuousTrainer:
    """Trainer with continuous visual feedback and real-time model updates."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
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
        
        # Visual state
        self.current_best_agent = None
        self.new_best_available = False
        self.visual_running = True
        
        # Create directories
        self.model_dir = os.path.join(config.training.model_dir, config.training.experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"Continuous Trainer initialized for experiment: {config.training.experiment_name}")
        print(f"Environment: {config.environment.name}")
        print(f"Agent: {config.agent.name}")
        print(f"Device: {self.agent.device}")
        print("🎬 Continuous visual display with real-time model updates!")
    
    def _create_environment(self, render_mode=None):
        """Create environment based on configuration."""
        env_type = self.config.environment.type
        params = self.config.environment.params
        
        if env_type == "salp_snake":
            # Debug print to check dimensions
            print(f"Creating environment with dimensions: {self.config.environment.width}x{self.config.environment.height}")
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
        """Main training loop with continuous visualization."""
        print(f"Starting continuous training for {self.config.training.max_episodes} episodes...")
        print("🎮 Visual display will run continuously and update with new best models!")
        print("Press Ctrl+C to stop training.")
        
        # Initialize current best agent
        self.current_best_agent = self._create_agent()
        
        # Start training thread (background)
        import threading
        training_thread = threading.Thread(target=self._background_training_loop, daemon=True)
        training_thread.start()
        
        # Run visual loop on main thread (required for macOS)
        self._continuous_visual_loop()
    def _background_training_loop(self):
        """Background training loop that runs in a separate thread."""
        start_time = time.time()
        
        try:
            for episode in range(self.config.training.max_episodes):
                if not self.visual_running:
                    print("Training stopped...")
                    break
                
                self.episode = episode
                
                # Run training episode
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
                    
                    # Check for new best model
                    if eval_metrics['mean_score'] > self.best_eval_score:
                        self.best_eval_score = eval_metrics['mean_score']
                        self._save_model("best_model.pth")
                        
                        # Update visual agent with new best model
                        self._update_visual_agent()
                        
                        print(f"🏆 NEW BEST MODEL! Score: {self.best_eval_score:.2f} - Visual updated!")
                
                # Save model periodically
                if episode % self.config.training.save_frequency == 0:
                    self._save_model(f"model_episode_{episode}.pth")
                
                # Print progress every episode
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{self.config.training.max_episodes}, "
                      f"Score: {episode_metrics['score']:.2f}, "
                      f"Food: {episode_metrics['food_collected']}, "
                      f"Steps: {episode_metrics['steps']}, "
                      f"Time: {elapsed_time:.1f}s")
        
        except Exception as e:
            print(f"Training error: {e}")
        
        finally:
            # Stop visual loop
            self.visual_running = False
            
            # Final save
            self._save_model("final_model.pth")
            self.logger.save_metrics()
            
            print("🎉 Training completed!")
            print(f"Best evaluation score: {self.best_eval_score:.2f}")
    
    def _continuous_visual_loop(self):
        """Continuous visual loop showing the current best agent."""
        print("🎬 Starting continuous visual display...")
        
        # Import pygame to ensure window focus
        try:
            import pygame
            # Force SDL to use a specific video driver for macOS compatibility
            os.environ['SDL_VIDEODRIVER'] = 'cocoa'
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            pygame.display.init()
        except ImportError:
            pass
        
        while self.visual_running:
            try:
                # Run visual episode with current best agent
                obs, _ = self.visual_env.reset()
                episode_reward = 0
                episode_steps = 0
                food_collected = 0
                
                done = False
                truncated = False
                
                print(f"🎮 Starting visual episode (Episode {self.episode})...")
                
                while not done and not truncated and episode_steps < self.config.training.max_steps_per_episode and self.visual_running:
                    # Render environment
                    self.visual_env.render()
                    
                    # Select action using current best agent
                    if self.current_best_agent is not None:
                        action = self.current_best_agent.select_action(obs, deterministic=True)
                    else:
                        # Use random action if no best agent yet
                        action = self.visual_env.action_space.sample()
                    
                    # Take step
                    obs, reward, done, truncated, info = self.visual_env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    food_collected = info.get('food_collected', 0)
                    
                    # Slower frame rate for better viewing
                    time.sleep(0.05)  # ~20 FPS for better visibility
                
                print(f"🎮 Visual episode completed: Score={episode_reward:.1f}, Food={food_collected}, Steps={episode_steps}")
                
                # Brief pause between episodes
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Visual loop error: {e}")
                import traceback
                traceback.print_exc()
                self.visual_running = False
                break
    
    def _update_visual_agent(self):
        """Update the visual agent with the current best model."""
        try:
            # Create new agent with same architecture
            new_visual_agent = self._create_agent()
            
            # Copy current agent's state
            new_visual_agent.policy.load_state_dict(self.agent.policy.state_dict())
            new_visual_agent.q1.load_state_dict(self.agent.q1.state_dict())
            new_visual_agent.q2.load_state_dict(self.agent.q2.state_dict())
            
            # Update current best agent
            self.current_best_agent = new_visual_agent
            
        except Exception as e:
            print(f"Error updating visual agent: {e}")
    
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
    """Demo continuous training script."""
    from config.base_config import get_sac_snake_config
    
    # Get configuration
    config = get_sac_snake_config()
    config.training.max_episodes = 500
    
    # Create continuous trainer
    trainer = ContinuousTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
