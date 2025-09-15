"""
Demo script for SALP RL experiments.
Provides easy access to training, evaluation, and environment testing.
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.base_config import get_sac_snake_config, get_sac_navigation_config, ExperimentConfig
from training.trainer import Trainer, EvaluationRunner
from training.live_trainer import LiveTrainer
from training.continuous_trainer import ContinuousTrainer
from environments.salp_snake_env import SalpSnakeEnv
import pygame


def demo_environment():
    """Demo the SALP Snake environment with manual control."""
    print("=== SALP Snake Environment Demo ===")
    print("Controls:")
    print("- HOLD SPACE: Inhale water")
    print("- RELEASE SPACE: Exhale water (thrust)")
    print("- ←/→ Arrow Keys: Steer nozzle")
    print("- Collect green food items!")
    print("- ESC: Quit")
    print()
    
    pygame.init()
    env = SalpSnakeEnv(render_mode="human", num_food_items=5)
    observation, info = env.reset()
    
    running = True
    nozzle_direction = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Inhale control
        inhale_control = 1.0 if keys[pygame.K_SPACE] else 0.0
        
        # Nozzle steering
        if keys[pygame.K_LEFT]:
            nozzle_direction = min(1.0, nozzle_direction + 0.03)
        elif keys[pygame.K_RIGHT]:
            nozzle_direction = max(-1.0, nozzle_direction - 0.03)
        
        # Apply action
        action = [inhale_control, nozzle_direction]
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if done or truncated:
            print(f"Episode ended! Score: {info['score']}, Food collected: {info['food_collected']}")
            observation, info = env.reset()
    
    env.close()


def train_agent(config_name: str = "sac_snake", episodes: Optional[int] = None, live: bool = False, continuous: bool = False):
    """Train a SALP agent."""
    if continuous:
        mode = "Continuous Visual Training"
    elif live:
        mode = "Live Training"
    else:
        mode = "Training"
    
    print(f"=== {mode} SALP Agent ({config_name}) ===")
    
    # Get configuration
    if config_name == "sac_snake":
        config = get_sac_snake_config()
    elif config_name == "sac_navigation":
        config = get_sac_navigation_config()
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Override episodes if specified
    if episodes is not None:
        config.training.max_episodes = episodes
    
    print(f"Configuration: {config_name}")
    print(f"Episodes: {config.training.max_episodes}")
    print(f"Environment: {config.environment.name}")
    print(f"Agent: {config.agent.name}")
    
    if continuous:
        print("🎮 Continuous visual display with real-time model updates!")
    elif live:
        print("🎬 Live visual episodes will show every 10 episodes!")
    print()
    
    # Create trainer and start training
    if continuous:
        trainer = ContinuousTrainer(config)
    elif live:
        trainer = LiveTrainer(config, visual_frequency=10)
    else:
        trainer = Trainer(config)
    trainer.train()


def evaluate_agent(model_path: str, config_name: str = "sac_snake", episodes: int = 5):
    """Evaluate a trained SALP agent."""
    print(f"=== Evaluating SALP Agent ===")
    print(f"Model: {model_path}")
    print(f"Configuration: {config_name}")
    print(f"Episodes: {episodes}")
    print()
    
    # Get configuration
    if config_name == "sac_snake":
        config = get_sac_snake_config()
    elif config_name == "sac_navigation":
        config = get_sac_navigation_config()
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create evaluation runner
    runner = EvaluationRunner(config, model_path)
    runner.run_episodes(episodes, deterministic=True)


def list_models(experiment_name: str = "salp_snake_sac"):
    """List available trained models."""
    print(f"=== Available Models for {experiment_name} ===")
    
    model_dir = os.path.join("models", experiment_name)
    
    if not os.path.exists(model_dir):
        print(f"No models found in {model_dir}")
        return
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth') and not f.endswith('_training_state.pth')]
    
    if not models:
        print("No model files found")
        return
    
    print("Available models:")
    for model in sorted(models):
        model_path = os.path.join(model_dir, model)
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"  - {model} ({size:.1f} MB)")
    
    print(f"\nTo evaluate a model, use:")
    print(f"python demo.py evaluate --model models/{experiment_name}/best_model.pth")


def quick_test():
    """Quick test to verify the implementation works."""
    print("=== Quick Implementation Test ===")
    
    try:
        # Test environment creation
        print("Testing environment creation...")
        env = SalpSnakeEnv(render_mode=None, num_food_items=3)
        obs, info = env.reset()
        print(f"✓ Environment created. Observation shape: {obs.shape}")
        
        # Test agent creation
        print("Testing agent creation...")
        config = get_sac_snake_config()
        from agents.sac_agent import SACAgent
        
        agent = SACAgent(
            config=config.agent,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_space=env.action_space
        )
        print(f"✓ Agent created. Device: {agent.device}")
        
        # Test action selection
        print("Testing action selection...")
        action = agent.select_action(obs, deterministic=True)
        print(f"✓ Action selected: {action}")
        
        # Test environment step
        print("Testing environment step...")
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step completed. Reward: {reward:.3f}")
        
        # Test replay buffer
        print("Testing replay buffer...")
        from core.base_agent import ReplayBuffer
        buffer = ReplayBuffer(1000, obs.shape[0], action.shape[0])
        buffer.add(obs, action, reward, next_obs, done)
        print(f"✓ Replay buffer working. Size: {len(buffer)}")
        
        env.close()
        print("\n✅ All tests passed! Implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function with command line interface."""
    parser = argparse.ArgumentParser(description="SALP RL Demo Script")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Environment demo
    env_parser = subparsers.add_parser('env', help='Demo the environment with manual control')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train a SALP agent')
    train_parser.add_argument('--config', default='sac_snake', choices=['sac_snake', 'sac_navigation'],
                             help='Configuration to use')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    train_parser.add_argument('--live', action='store_true', help='Show live visual episodes during training')
    train_parser.add_argument('--continuous', action='store_true', help='Continuous visual display with real-time model updates')
    
    # Evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', required=True, help='Path to model file')
    eval_parser.add_argument('--config', default='sac_snake', choices=['sac_snake', 'sac_navigation'],
                            help='Configuration to use')
    eval_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    
    # List models
    list_parser = subparsers.add_parser('list', help='List available trained models')
    list_parser.add_argument('--experiment', default='salp_snake_sac', help='Experiment name')
    
    # Quick test
    test_parser = subparsers.add_parser('test', help='Quick test of the implementation')
    
    args = parser.parse_args()
    
    if args.command == 'env':
        demo_environment()
    elif args.command == 'train':
        train_agent(args.config, args.episodes, getattr(args, 'live', False), getattr(args, 'continuous', False))
    elif args.command == 'evaluate':
        evaluate_agent(args.model, args.config, args.episodes)
    elif args.command == 'list':
        list_models(args.experiment)
    elif args.command == 'test':
        quick_test()
    else:
        # No command specified, show help and run quick test
        parser.print_help()
        print("\n" + "="*50)
        quick_test()


if __name__ == "__main__":
    main()
