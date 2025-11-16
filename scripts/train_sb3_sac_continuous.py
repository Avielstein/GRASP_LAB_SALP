"""
Continuous visual training with Stable Baselines3 SAC.

Shows the agent playing continuously while training in the background.
The visual display automatically updates when better models are found.

This mimics the original continuous_trainer.py but uses SB3 SAC.
"""

import os
import sys
import time
import threading
import argparse
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config


class ContinuousSB3Trainer:
    """
    Continuous visual training with SB3 SAC.
    
    - Training runs in background thread
    - Visual display runs continuously on main thread (required for macOS)
    - Visual agent updates automatically when better model is found
    """
    
    def __init__(self, total_timesteps: int = 100000, eval_freq: int = 5000, single_food: bool = False):
        # Use single food config if requested, otherwise default multi-food
        if single_food:
            from config.base_config import get_single_food_optimal_config
            self.config = get_single_food_optimal_config()
            print("🎯 Using SINGLE FOOD configuration")
        else:
            self.config = get_sac_snake_config()
            print("🍎 Using MULTI FOOD configuration")
        
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        
        # Create environments
        print("Creating environments...")
        self.train_env = SalpSnakeEnv(render_mode=None, **self.config.environment.params)
        self.eval_env = SalpSnakeEnv(render_mode=None, **self.config.environment.params)
        self.visual_env = SalpSnakeEnv(render_mode="human", **self.config.environment.params)
        print("✓ Environments created")
        
        # Create agent
        print("Creating SB3 SAC agent...")
        self.agent = SAC(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=self.config.agent.learning_rate,
            buffer_size=self.config.agent.buffer_size,
            batch_size=self.config.agent.batch_size,
            tau=self.config.agent.tau,
            gamma=self.config.agent.gamma,
            policy_kwargs=dict(net_arch=self.config.agent.hidden_sizes),
            verbose=0,  # Quiet mode for cleaner output
        )
        print("✓ Agent created")
        
        # Current best model for visualization
        self.visual_model = None
        self.best_mean_reward = -float('inf')
        
        # Training state
        self.training_active = True
        self.model_updated = False
        self.current_timesteps = 0
        
        # Save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"models/sb3_sac_continuous_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"✓ Save directory: {self.save_dir}")
    
    def _format_time(self, seconds):
        """Format time in readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        else:
            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
    def _evaluate_model(self, model, n_episodes=5):
        """Evaluate model performance."""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def _training_loop(self):
        """Background training loop."""
        print(f"\n{'=' * 70}")
        print(f"🚀 Starting training for {self.total_timesteps:,} timesteps")
        print(f"   Evaluation frequency: every {self.eval_freq} steps")
        print(f"{'=' * 70}\n")
        
        start_time = time.time()
        
        # Custom callback for evaluation
        class EvalAndUpdateCallback(BaseCallback):
            def __init__(self, trainer, eval_freq):
                super().__init__()
                self.trainer = trainer
                self.eval_freq = eval_freq
                self.n_evals = 0
                
            def _on_step(self):
                # Evaluate periodically
                if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                    mean_reward, std_reward = self.trainer._evaluate_model(self.model)
                    self.n_evals += 1
                    
                    elapsed = time.time() - start_time
                    time_str = self.trainer._format_time(elapsed)
                    
                    print(f"\n📊 Evaluation #{self.n_evals} (Step {self.n_calls:,}, Time: {time_str})")
                    print(f"   Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
                    
                    # Check if this is a new best
                    if mean_reward > self.trainer.best_mean_reward:
                        self.trainer.best_mean_reward = mean_reward
                        
                        # Save best model
                        best_path = os.path.join(self.trainer.save_dir, "best_model")
                        self.model.save(best_path)
                        
                        # Update visual model
                        self.trainer.visual_model = SAC.load(best_path, env=self.trainer.visual_env)
                        self.trainer.model_updated = True
                        
                        print(f"   🏆 NEW BEST MODEL! (Previous best: {self.trainer.best_mean_reward - mean_reward:.1f})")
                        print(f"   ✓ Model saved and visual display updated!")
                    else:
                        print(f"   (Current best: {self.trainer.best_mean_reward:.1f})")
                    
                    self.trainer.current_timesteps = self.n_calls
                
                return True
        
        try:
            callback = EvalAndUpdateCallback(self, self.eval_freq)
            self.agent.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                log_interval=None  # Disable default logging
            )
            
            print(f"\n{'=' * 70}")
            print("✅ Training completed successfully!")
            print(f"   Best mean reward: {self.best_mean_reward:.1f}")
            print(f"   Total time: {self._format_time(time.time() - start_time)}")
            print(f"{'=' * 70}\n")
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Save final model
            final_path = os.path.join(self.save_dir, "final_model")
            self.agent.save(final_path)
            print(f"✓ Final model saved to: {final_path}")
            
            self.training_active = False
    
    def _visual_loop(self):
        """Continuous visual display loop (runs on main thread for macOS compatibility)."""
        print(f"\n🎬 Starting continuous visual display...")
        print("   The agent will play continuously while training in background")
        print("   Visual will update automatically when better models are found\n")
        
        # Initialize pygame for macOS
        try:
            import pygame
            os.environ['SDL_VIDEODRIVER'] = 'cocoa'
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
            pygame.init()
            pygame.display.init()
        except:
            pass
        
        episode_count = 0
        
        while self.training_active:
            try:
                # Use visual model if available, otherwise random actions
                obs, _ = self.visual_env.reset()
                done = False
                episode_reward = 0
                episode_steps = 0
                
                # Show current status
                if self.visual_model is not None:
                    if self.model_updated:
                        print(f"🎬 Episode {episode_count + 1} - UPDATED MODEL (Step {self.current_timesteps:,}, Best: {self.best_mean_reward:.1f})")
                        self.model_updated = False
                    else:
                        print(f"🎬 Episode {episode_count + 1} - Current best model (Step {self.current_timesteps:,})")
                else:
                    print(f"🎬 Episode {episode_count + 1} - Random exploration (waiting for first model...)")
                
                while not done and self.training_active:
                    # Render
                    self.visual_env.render()
                    
                    # Get action
                    if self.visual_model is not None:
                        action, _ = self.visual_model.predict(obs, deterministic=True)
                    else:
                        action = self.visual_env.action_space.sample()
                    
                    # Step
                    obs, reward, terminated, truncated, info = self.visual_env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    # Slower for better viewing
                    time.sleep(0.03)  # ~30 FPS
                
                # Episode summary
                food = info.get('food_collected', 0)
                print(f"   Reward: {episode_reward:.1f}, Food: {food}, Steps: {episode_steps}\n")
                
                episode_count += 1
                
                # Brief pause between episodes
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Visual loop error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Clean up
        self.visual_env.close()
        print("🎬 Visual display closed")
    
    def train(self):
        """Start training with continuous visualization."""
        print("=" * 70)
        print("SB3 SAC Training with Continuous Visualization")
        print("=" * 70)
        
        # Start training in background thread
        training_thread = threading.Thread(target=self._training_loop, daemon=True)
        training_thread.start()
        
        # Give training a moment to start
        time.sleep(2)
        
        # Run visual loop on main thread (required for macOS pygame)
        try:
            self._visual_loop()
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            self.training_active = False
        
        # Wait for training thread to finish
        training_thread.join(timeout=5)
        
        print(f"\n{'=' * 70}")
        print(f"Training session complete!")
        print(f"Models saved in: {self.save_dir}")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train SB3 SAC with continuous visualization"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N steps (default: 5000)"
    )
    parser.add_argument(
        "--single-food",
        action="store_true",
        help="Use single food configuration (default: multi-food)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    if args.single_food:
        print("🎯 SINGLE FOOD LEARNING - Agent learns to reach one target")
    else:
        print("🍎 MULTI FOOD LEARNING - Agent learns to collect multiple targets")
    print("=" * 70 + "\n")
    
    trainer = ContinuousSB3Trainer(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        single_food=args.single_food
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
