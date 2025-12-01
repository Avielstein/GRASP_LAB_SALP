#!/usr/bin/env python3
"""
Record a trained SALP model and create MP4 and GIF visualizations.

Usage:
    python record_model.py
    python record_model.py --timesteps 5000
    python record_model.py --model path/to/model.pth
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Direct imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch.nn as nn
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo
import subprocess


class Actor(nn.Module):
    """Actor network matching the saved checkpoint structure."""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        all_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            all_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        all_layers.append(nn.Linear(prev_size, action_dim * 2))
        self.layers = nn.ModuleList(all_layers)
        
    def forward(self, obs, deterministic=False):
        x = obs
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = self.layers[-1](x)
        mean, log_std = output.chunk(2, dim=-1)
        
        if deterministic:
            return torch.tanh(mean)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action)


def record_model(model_path: str, max_timesteps: int = 5000, watch: bool = True):
    """Record a model running."""
    from salp.environments.salp_snake_env import SalpSnakeEnv
    import pygame
    
    print("=" * 70)
    print("SALP MODEL RECORDING")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Single episode: {max_timesteps} steps")
    print(f"Watch live: {watch}")
    print("=" * 70 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("assets/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = f"single_food_5000steps_{timestamp}"
    
    # Create environment - use "human" mode if watching, otherwise "rgb_array" 
    render_mode = "human" if watch else "rgb_array"
    
    base_env = SalpSnakeEnv(
        render_mode=render_mode,
        num_food_items=1,
        food_reward=1000.0,
        collision_penalty=-10.0,
        time_penalty=-0.1,
        proximity_reward_weight=5.0,
        respawn_food=True,  # Food respawns - continuous navigation!
        forced_breathing=True,
        max_steps_without_food=10000  # High limit so episode doesn't end early
    )
    
    # Only wrap with RecordVideo if not in human mode
    if not watch:
        env = RecordVideo(
            env=base_env,
            video_folder=str(output_dir),
            episode_trigger=lambda x: True,
            name_prefix=video_name,
            disable_logger=True
        )
    else:
        env = base_env
        print("⚠️  Note: Visual rendering enabled - video will NOT be saved")
        print("    Run without --watch to save MP4/GIF\n")
    
    print(f"Environment created")
    print(f"Recording to: {output_dir}/{video_name}\n")
    
    # Load model
    print(f"Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(obs_dim, action_dim, hidden_sizes=[256, 256])
    actor.load_state_dict(checkpoint['policy_state_dict'])
    actor.eval()
    
    print("✓ Model loaded\n")
    
    # Single long episode
    obs, _ = env.reset()
    total_reward = 0
    total_food = 0
    
    print(f"🎬 Recording single {max_timesteps}-step episode...")
    print(f"   (Food respawns each time collected)\n")
    
    for step in range(max_timesteps):
        # Get action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = actor(obs_tensor, deterministic=True)
            action = action_tensor.squeeze(0).numpy()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_food = info.get('food_collected', 0)
        
        # Render if in human mode
        if watch:
            env.render()
            import time
            time.sleep(0.016)  # ~60 FPS
        
        # Print progress
        if (step + 1) % 500 == 0:
            print(f"   Step {step + 1}/{max_timesteps} - Reward: {total_reward:.1f}, Food: {total_food}")
        
        # Check for early termination (collision)
        if terminated:
            print(f"\n⚠️  Episode ended early at step {step + 1} (collision)")
            break
    
    env.close()
    
    print(f"\n✓ Recording complete!")
    print(f"   Total steps: {step + 1}")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Total food collected: {total_food}\n")
    
    # Find the video file (only if not watching live)
    if not watch:
        video_files = list(output_dir.glob(f"{video_name}*.mp4"))
    else:
        video_files = []
    
    if video_files and not watch:
        video_file = video_files[0]
        print(f"📹 Video saved: {video_file}")
        print(f"   Size: {video_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Convert to GIF
        print(f"\n🎨 Converting to GIF...")
        gif_file = video_file.with_suffix('.gif')
        
        try:
            # Use ffmpeg to convert (with compression)
            subprocess.run([
                'ffmpeg', '-i', str(video_file),
                '-vf', 'fps=10,scale=400:-1:flags=lanczos',
                '-y', str(gif_file)
            ], check=True, capture_output=True)
            
            print(f"✓ GIF created: {gif_file}")
            print(f"   Size: {gif_file.stat().st_size / 1024 / 1024:.1f} MB")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  GIF conversion failed: {e}")
            print("   (ffmpeg may not be installed)")
        except FileNotFoundError:
            print("⚠️  ffmpeg not found - cannot create GIF")
            print("   Install with: brew install ffmpeg")
        
        return video_file, gif_file if gif_file.exists() else None
    elif watch:
        print("✓ Live viewing complete (no video saved)")
        return None, None
    else:
        print("⚠️  No video file created")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Record SALP model")
    
    parser.add_argument(
        '--model',
        default='data/models/single_food_optimal_navigation/best_model.pth',
        help='Path to model'
    )
    
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=5000,
        help='Number of timesteps to record'
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch live (no video saved). Without this flag, saves MP4/GIF but no display.'
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    video_file, gif_file = record_model(args.model, args.timesteps, watch=args.watch)
    
    print("\n" + "=" * 70)
    print("RECORDING COMPLETE!")
    print("=" * 70)
    if video_file:
        print(f"MP4: {video_file}")
    if gif_file:
        print(f"GIF: {gif_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
