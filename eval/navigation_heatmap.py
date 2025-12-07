#!/usr/bin/env python3
"""
Navigation Heatmap Visualization
Runs multiple trials of point-to-point navigation and generates a heatmap
showing path consistency and navigation performance.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from src.salp.environments.salp_snake_env import SalpSnakeEnv


class NavigationHeatmapEvaluator:
    """Evaluates navigation performance across multiple trials and generates heatmaps."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 start_pos: Tuple[float, float] = (100, 300),
                 goal_pos: Tuple[float, float] = (700, 300),
                 env_width: int = 800,
                 env_height: int = 600):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model (.zip for SB3)
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            env_width: Environment width in pixels
            env_height: Environment height in pixels
        """
        self.model_path = model_path
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.env_width = env_width
        self.env_height = env_height
        self.goal_radius = 50  # Success threshold
        
        # Calculate optimal (straight-line) distance
        self.optimal_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # Load model if provided
        self.model = None
        if model_path:
            print(f"Loading model from: {model_path}")
            self.model = SAC.load(model_path)
            print("✓ Model loaded successfully")
    
    def create_environment(self) -> SalpSnakeEnv:
        """Create a clean navigation environment."""
        env = SalpSnakeEnv(
            render_mode=None,
            width=self.env_width,
            height=self.env_height,
            num_food_items=1,  # Single food target at goal
            forced_breathing=True,
            respawn_food=False,  # Don't respawn after collection
            max_steps_without_food=3000  # Longer timeout for navigation
        )
        return env
    
    def run_single_trial(self, env: SalpSnakeEnv, max_steps: int = 1000,
                        use_random: bool = False) -> Dict:
        """
        Run a single navigation trial.
        
        Returns:
            Dictionary containing trajectory and metrics
        """
        # Reset environment and set custom start position
        obs, _ = env.reset()
        env.robot_pos = self.start_pos.copy()
        env.robot_velocity = np.array([0.0, 0.0], dtype=float)
        
        # Randomize initial orientation (model should handle any direction)
        env.robot_angle = np.random.uniform(-np.pi, np.pi)
        env.robot_angular_velocity = 0.0
        
        # Place food at goal position (this is what the model was trained to navigate to!)
        env.food_positions = [self.goal_pos.copy()]
        
        # Reset steps counter to allow navigation
        env.steps_since_food = 0
        
        # Update observation to reflect new position and food location
        obs = env._get_extended_observation()
        
        # Record trajectory
        positions = [self.start_pos.copy()]
        velocities = []
        actions_taken = []
        
        done = False
        truncated = False
        steps = 0
        
        while steps < max_steps:
            # Select action
            if use_random or self.model is None:
                action = env.action_space.sample()
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            
            actions_taken.append(action.copy())
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            # Log if environment tried to end episode (but we'll continue anyway)
            if (done or truncated) and steps < 100:
                print(f"\n  ℹ️  Environment signaled termination at step {steps} (ignoring)")
                print(f"    done={done}, truncated={truncated}")
                if info.get('collision'):
                    print(f"    Collision flag: {info.get('collision')}")
            
            # Record state
            positions.append(env.robot_pos.copy())
            velocities.append(env.robot_velocity.copy())
            
            steps += 1
            
            # Check if goal reached (ONLY termination condition we respect)
            distance_to_goal = np.linalg.norm(env.robot_pos - self.goal_pos)
            if distance_to_goal < self.goal_radius:
                break  # Success! End trial
        
        # Calculate metrics for this trial
        positions_array = np.array(positions)
        
        # Remove consecutive duplicate positions (robot not moving)
        # This is important because robot may stay in place for first few steps
        unique_positions = [positions_array[0]]
        for i in range(1, len(positions_array)):
            if not np.array_equal(positions_array[i], positions_array[i-1]):
                unique_positions.append(positions_array[i])
        unique_positions_array = np.array(unique_positions)
        
        # Path length (cumulative distance) using only unique positions
        path_length_to_stop = np.sum(np.linalg.norm(np.diff(unique_positions_array, axis=0), axis=1))
        
        # Final distance to goal
        final_distance = np.linalg.norm(positions_array[-1] - self.goal_pos)
        
        # Add the remaining distance to goal center (robot stopped within radius)
        # This gives us the full path length as if it reached the goal center
        path_length = path_length_to_stop + final_distance
        
        # Success (reached within radius)
        success = final_distance < self.goal_radius
        
        # Path efficiency ratio
        path_ratio = path_length / self.optimal_distance if self.optimal_distance > 0 else float('inf')
        
        # Straightness index (euclidean / path_length)
        straightness = self.optimal_distance / path_length if path_length > 0 else 0
        
        # Mean lateral deviation from optimal path
        lateral_deviation = self._calculate_lateral_deviation(positions_array)
        
        # Calculate area covered (bounding box)
        x_min, y_min = positions_array.min(axis=0)
        x_max, y_max = positions_array.max(axis=0)
        area_covered = (x_max - x_min) * (y_max - y_min)
        
        # Optimal area (minimal bounding box for straight line)
        # For a straight line, the "area" would just be length × 0, but practically it's the line swept area
        # We'll use a minimal thickness estimate based on goal radius
        optimal_area = self.optimal_distance * (self.goal_radius * 2)
        area_ratio = area_covered / optimal_area if optimal_area > 0 else float('inf')
        
        return {
            'positions': positions_array,
            'velocities': velocities,
            'actions': actions_taken,
            'steps': steps,
            'path_length': path_length,
            'path_ratio': path_ratio,
            'straightness': straightness,
            'final_distance': final_distance,
            'success': success,
            'lateral_deviation': lateral_deviation,
            'area_covered': area_covered,
            'area_ratio': area_ratio,
            'x_range': x_max - x_min,
            'y_range': y_max - y_min
        }
    
    def _calculate_lateral_deviation(self, positions: np.ndarray) -> float:
        """Calculate mean perpendicular distance from optimal straight line."""
        if len(positions) < 2:
            return 0.0
        
        # Vector from start to goal
        optimal_vector = self.goal_pos - self.start_pos
        optimal_length = np.linalg.norm(optimal_vector)
        
        if optimal_length == 0:
            return 0.0
        
        optimal_unit = optimal_vector / optimal_length
        
        # Calculate perpendicular distance for each position
        deviations = []
        for pos in positions:
            # Vector from start to current position
            pos_vector = pos - self.start_pos
            
            # Project onto optimal line
            projection_length = np.dot(pos_vector, optimal_unit)
            projection = self.start_pos + projection_length * optimal_unit
            
            # Perpendicular distance
            deviation = np.linalg.norm(pos - projection)
            deviations.append(deviation)
        
        return np.mean(deviations)
    
    def run_multiple_trials(self, num_trials: int = 30, 
                          use_random: bool = False,
                          max_steps: int = 1000) -> List[Dict]:
        """Run multiple navigation trials."""
        print(f"\n{'='*60}")
        print(f"Running {num_trials} navigation trials...")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"Optimal distance: {self.optimal_distance:.1f} px")
        print(f"{'='*60}\n")
        
        env = self.create_environment()
        results = []
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}...", end=' ')
            
            trial_result = self.run_single_trial(env, max_steps=max_steps, use_random=use_random)
            results.append(trial_result)
            
            status = "✓ SUCCESS" if trial_result['success'] else "✗ FAILED"
            
            # Add warning for very short trials
            warning = ""
            if trial_result['steps'] < 300 and not trial_result['success']:
                warning = " ⚠️ (early stop)"
            
            print(f"{status} | Steps: {trial_result['steps']} | "
                  f"Path ratio: {trial_result['path_ratio']:.2f} | "
                  f"Final dist: {trial_result['final_distance']:.1f}px{warning}")
            
            # Debug: show first and last few positions
            if i == 0:  # Only for first trial
                positions = trial_result['positions']
                print(f"  First 5 positions: {positions[:5]}")
                print(f"  Last 5 positions: {positions[-5:]}")
                print(f"  Total unique positions: {len(np.unique(positions, axis=0))}")
        
        env.close()
        return results
    
    def generate_heatmap(self, results: List[Dict], 
                        output_path: str = None,
                        compare_results: Optional[List[Dict]] = None):
        """Generate heatmap visualization from multiple trials."""
        
        # Collect all positions from all trials
        all_positions = []
        for result in results:
            all_positions.extend(result['positions'])
        all_positions = np.array(all_positions)
        
        # Calculate statistics
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_path_length = np.mean([r['path_length'] for r in results])
        std_path_length = np.std([r['path_length'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_straightness = np.mean([r['straightness'] for r in results])
        std_straightness = np.std([r['straightness'] for r in results])
        avg_lateral_dev = np.mean([r['lateral_deviation'] for r in results])
        
        # Create figure
        if compare_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            axes = [ax1, ax2]
            titles = ['Trained Agent', 'Random Policy']
            results_list = [results, compare_results]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            axes = [ax]
            titles = ['Navigation Heatmap']
            results_list = [results]
        
        for idx, (ax, title, trial_results) in enumerate(zip(axes, titles, results_list)):
            # Collect positions for this set of results
            positions = []
            for result in trial_results:
                positions.extend(result['positions'])
            positions = np.array(positions)
            
            # Create 2D histogram (heatmap)
            bins_x = 100
            bins_y = int(bins_x * self.env_height / self.env_width)
            
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=[bins_x, bins_y],
                range=[[0, self.env_width], [0, self.env_height]]
            )
            
            # Apply gaussian smoothing
            heatmap = gaussian_filter(heatmap, sigma=1.5)
            
            # Plot heatmap
            extent = [0, self.env_width, 0, self.env_height]
            
            # Custom colormap: blue (low) -> yellow -> red (high)
            colors = ['#0a0a2e', '#1e3a8a', '#3b82f6', '#fbbf24', '#f97316', '#dc2626']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('navigation', colors, N=n_bins)
            
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                          cmap=cmap, aspect='auto', alpha=0.8, interpolation='bilinear')
            
            # Draw optimal path (dashed line)
            ax.plot([self.start_pos[0], self.goal_pos[0]], 
                   [self.start_pos[1], self.goal_pos[1]], 
                   'w--', linewidth=3, alpha=0.8, label='Optimal Path')
            
            # Draw start and goal
            ax.plot(self.start_pos[0], self.start_pos[1], 'go', 
                   markersize=20, markeredgecolor='darkgreen', 
                   markeredgewidth=3, label='Start', zorder=10)
            ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', 
                   markersize=25, markeredgecolor='darkred', 
                   markeredgewidth=2, label='Goal', zorder=10)
            
            # Draw goal radius circle
            goal_circle = plt.Circle(self.goal_pos, self.goal_radius, 
                                    color='red', fill=False, 
                                    linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(goal_circle)
            
            # Draw tank boundaries
            tank_margin = 50
            boundary = patches.Rectangle((tank_margin, tank_margin), 
                                        self.env_width - 2*tank_margin, 
                                        self.env_height - 2*tank_margin,
                                        linewidth=2, edgecolor='cyan', 
                                        facecolor='none', alpha=0.3)
            ax.add_patch(boundary)
            
            # Calculate stats for this set
            stats_success_rate = sum(r['success'] for r in trial_results) / len(trial_results)
            stats_avg_path = np.mean([r['path_length'] for r in trial_results])
            stats_std_path = np.std([r['path_length'] for r in trial_results])
            stats_avg_straight = np.mean([r['straightness'] for r in trial_results])
            stats_std_straight = np.std([r['straightness'] for r in trial_results])
            
            # Add text with statistics
            stats_text = (
                f"Trials: {len(trial_results)}\n"
                f"Success Rate: {stats_success_rate*100:.1f}%\n"
                f"Avg Path Length: {stats_avg_path:.0f} ± {stats_std_path:.0f} px\n"
                f"Straightness: {stats_avg_straight:.3f} ± {stats_std_straight:.3f}\n"
                f"Optimal Distance: {self.optimal_distance:.0f} px"
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white', family='monospace')
            
            ax.set_xlim(0, self.env_width)
            ax.set_ylim(0, self.env_height)
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title(f'{title}\n(N={len(trial_results)} trials)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Visit Density', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eval/results/navigation_heatmap_{timestamp}.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Heatmap saved to: {output_path}")
        
        plt.close()
        
        return output_path
    
    def plot_trajectories(self, results: List[Dict], output_path: str = None):
        """Plot individual trajectories as lines (easier to see than heatmap)."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Draw tank boundaries
        tank_margin = 50
        boundary = patches.Rectangle((tank_margin, tank_margin), 
                                    self.env_width - 2*tank_margin, 
                                    self.env_height - 2*tank_margin,
                                    linewidth=2, edgecolor='black', 
                                    facecolor='lightgray', alpha=0.2)
        ax.add_patch(boundary)
        
        # Draw optimal path (dashed line)
        ax.plot([self.start_pos[0], self.goal_pos[0]], 
               [self.start_pos[1], self.goal_pos[1]], 
               'k--', linewidth=2, alpha=0.5, label='Optimal Path', zorder=1)
        
        # Plot each trajectory with a different color
        colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
        
        for i, (result, color) in enumerate(zip(results, colors)):
            positions = result['positions']
            
            # Plot trajectory line
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=color, linewidth=1.5, alpha=0.7, 
                   label=f"Trial {i+1} ({result['steps']} steps)")
            
            # Mark end position
            if result['success']:
                marker = '*'
                markersize = 15
                markeredgecolor = 'gold'
            else:
                marker = 'x'
                markersize = 10
                markeredgecolor = 'darkred'
            
            ax.plot(positions[-1, 0], positions[-1, 1], 
                   marker=marker, markersize=markersize, 
                   color=color, markeredgecolor=markeredgecolor,
                   markeredgewidth=2, zorder=10)
        
        # Draw start and goal
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', 
               markersize=20, markeredgecolor='darkgreen', 
               markeredgewidth=3, label='Start', zorder=15)
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', 
               markersize=30, markeredgecolor='darkred', 
               markeredgewidth=3, label='Goal', zorder=15)
        
        # Draw goal radius circle
        goal_circle = plt.Circle(self.goal_pos, self.goal_radius, 
                                color='red', fill=False, 
                                linestyle='--', linewidth=2, alpha=0.4)
        ax.add_patch(goal_circle)
        
        # Calculate stats
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_path_length = np.mean([r['path_length'] for r in results])
        std_path_length = np.std([r['path_length'] for r in results])
        
        # Add title and labels
        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_height)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.set_title(f'Individual Navigation Trajectories (N={len(results)})\n'
                    f'Success Rate: {success_rate*100:.1f}% | '
                    f'Avg Path: {avg_path_length:.0f}±{std_path_length:.0f}px', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend (outside plot)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eval/results/navigation_trajectories_{timestamp}.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Trajectory plot saved to: {output_path}")
        
        plt.close()
        
        return output_path
    
    def save_statistics(self, results: List[Dict], output_path: str = None):
        """Save detailed statistics to JSON file."""
        stats = {
            'num_trials': int(len(results)),
            'start_position': self.start_pos.tolist(),
            'goal_position': self.goal_pos.tolist(),
            'optimal_distance': float(self.optimal_distance),
            'success_rate': float(sum(r['success'] for r in results) / len(results)),
            'successful_trials': int(sum(r['success'] for r in results)),
            'failed_trials': int(len(results) - sum(r['success'] for r in results)),
            'avg_path_length': float(np.mean([r['path_length'] for r in results])),
            'std_path_length': float(np.std([r['path_length'] for r in results])),
            'avg_path_ratio': float(np.mean([r['path_ratio'] for r in results])),
            'avg_straightness': float(np.mean([r['straightness'] for r in results])),
            'std_straightness': float(np.std([r['straightness'] for r in results])),
            'avg_steps': float(np.mean([r['steps'] for r in results])),
            'std_steps': float(np.std([r['steps'] for r in results])),
            'avg_lateral_deviation': float(np.mean([r['lateral_deviation'] for r in results])),
            'std_lateral_deviation': float(np.std([r['lateral_deviation'] for r in results])),
            'avg_final_distance': float(np.mean([r['final_distance'] for r in results])),
            'avg_area_covered': float(np.mean([r['area_covered'] for r in results])),
            'std_area_covered': float(np.std([r['area_covered'] for r in results])),
            'avg_area_ratio': float(np.mean([r['area_ratio'] for r in results])),
            'avg_x_range': float(np.mean([r['x_range'] for r in results])),
            'avg_y_range': float(np.mean([r['y_range'] for r in results])),
        }
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eval/results/navigation_stats_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Statistics saved to: {output_path}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Navigation Heatmap Evaluation')
    parser.add_argument('--model', type=str, help='Path to trained model (.zip)')
    parser.add_argument('--trials', type=int, default=30, help='Number of trials to run')
    parser.add_argument('--max-steps', type=int, default=3000, help='Max steps per trial')
    parser.add_argument('--start', type=str, default='100,300', 
                       help='Start position as x,y')
    parser.add_argument('--goal', type=str, default='700,300', 
                       help='Goal position as x,y')
    parser.add_argument('--compare-random', action='store_true',
                       help='Compare with random policy baseline')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--plot-trajectories', action='store_true',
                       help='Plot individual trajectories instead of heatmap')
    
    args = parser.parse_args()
    
    # Parse positions
    start_pos = tuple(map(float, args.start.split(',')))
    goal_pos = tuple(map(float, args.goal.split(',')))
    
    # Create evaluator
    evaluator = NavigationHeatmapEvaluator(
        model_path=args.model,
        start_pos=start_pos,
        goal_pos=goal_pos
    )
    
    # Run trials with trained model (or random if no model)
    use_random = args.model is None
    results = evaluator.run_multiple_trials(
        num_trials=args.trials,
        use_random=use_random,
        max_steps=args.max_steps
    )
    
    # Optionally compare with random policy
    compare_results = None
    if args.compare_random and args.model is not None:
        print("\n" + "="*60)
        print("Running random policy baseline for comparison...")
        print("="*60)
        evaluator_random = NavigationHeatmapEvaluator(
            model_path=None,
            start_pos=start_pos,
            goal_pos=goal_pos
        )
        compare_results = evaluator_random.run_multiple_trials(
            num_trials=args.trials,
            use_random=True
        )
    
    # Generate visualization
    print("\n" + "="*60)
    if args.plot_trajectories:
        print("Generating trajectory plot...")
        print("="*60)
        viz_path = evaluator.plot_trajectories(results, output_path=args.output)
    else:
        print("Generating heatmap visualization...")
        print("="*60)
        viz_path = evaluator.generate_heatmap(
            results, 
            output_path=args.output,
            compare_results=compare_results
        )
    
    # Save statistics
    stats = evaluator.save_statistics(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model if args.model else 'Random Policy'}")
    print(f"Trials: {stats['num_trials']}")
    print(f"Success Rate: {stats['success_rate']*100:.1f}% "
          f"({stats['successful_trials']}/{stats['num_trials']})")
    print(f"\nPath Metrics:")
    print(f"  Avg Path Length: {stats['avg_path_length']:.1f} ± {stats['std_path_length']:.1f} px")
    print(f"  Path Efficiency: {stats['avg_path_ratio']:.2f}x optimal")
    print(f"  Straightness: {stats['avg_straightness']:.3f} ± {stats['std_straightness']:.3f}")
    print(f"\nArea Coverage:")
    print(f"  Avg Area Covered: {stats['avg_area_covered']:.0f} ± {stats['std_area_covered']:.0f} px²")
    print(f"  Area Ratio: {stats['avg_area_ratio']:.2f}x optimal")
    print(f"  Avg X Range: {stats['avg_x_range']:.1f} px")
    print(f"  Avg Y Range: {stats['avg_y_range']:.1f} px")
    print(f"\nOther Metrics:")
    print(f"  Avg Steps: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
    print(f"  Lateral Deviation: {stats['avg_lateral_deviation']:.1f} ± {stats['std_lateral_deviation']:.1f} px")
    print("="*60)
    
    print(f"\n✅ Evaluation complete! Check output: {viz_path}")


if __name__ == "__main__":
    main()
