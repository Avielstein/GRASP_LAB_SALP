"""
Custom TensorBoard callback for detailed metrics logging.

This callback logs comprehensive metrics during training including:
- Navigation performance (success rate, path efficiency)
- Reward component breakdown
- Physical metrics (actions, velocities)
- Diagnostic metrics (stalling, oscillation)

See METRICS.md for detailed documentation of all logged metrics.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Any
from collections import deque


class DetailedMetricsCallback(BaseCallback):
    """
    Custom callback for logging detailed metrics to TensorBoard.
    
    Tracks episode-level metrics and logs aggregated statistics periodically.
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """
        Args:
            log_freq: Frequency (in steps) to log aggregated metrics
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episode_truncations = deque(maxlen=100)
        
        # Navigation metrics
        self.episode_path_lengths = deque(maxlen=100)
        self.episode_direct_distances = deque(maxlen=100)
        self.episode_efficiencies = deque(maxlen=100)
        self.episode_final_distances = deque(maxlen=100)
        self.episode_initial_distances = deque(maxlen=100)
        
        # Action statistics
        self.episode_compressions = deque(maxlen=100)
        self.episode_coast_times = deque(maxlen=100)
        self.episode_nozzle_angles = deque(maxlen=100)
        
        # Velocity statistics
        self.episode_velocities = deque(maxlen=100)
        
        # Reward components
        self.episode_r_tracks = deque(maxlen=100)
        self.episode_r_headings = deque(maxlen=100)
        self.episode_r_cycles = deque(maxlen=100)
        self.episode_r_energies = deque(maxlen=100)
        self.episode_r_smooths = deque(maxlen=100)
    
    def _on_step(self) -> bool:
        """
        Called at each environment step.
        
        Collects metrics from completed episodes and logs aggregated statistics.
        """
        # Check if any episodes finished
        infos = self.locals.get('infos', [])
        
        for idx, info in enumerate(infos):
            if 'episode' in info:
                # Episode just finished - extract metrics
                ep_info = info['episode']
                
                # Standard metrics
                ep_reward = ep_info['r']
                ep_length = ep_info['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Success/truncation
                terminated = self.locals['dones'][idx] if 'dones' in self.locals else False
                truncated = info.get('TimeLimit.truncated', False)
                self.episode_successes.append(1.0 if terminated and not truncated else 0.0)
                self.episode_truncations.append(1.0 if truncated else 0.0)
                
                # Navigation metrics (from environment)
                if 'path_length' in info:
                    self.episode_path_lengths.append(info['path_length'])
                if 'direct_distance' in info:
                    self.episode_direct_distances.append(info['direct_distance'])
                if 'path_efficiency' in info:
                    self.episode_efficiencies.append(info['path_efficiency'])
                if 'final_distance' in info:
                    self.episode_final_distances.append(info['final_distance'])
                if 'initial_distance' in info:
                    self.episode_initial_distances.append(info['initial_distance'])
                
                # Action statistics
                if 'avg_compression' in info:
                    self.episode_compressions.append(info['avg_compression'])
                if 'avg_coast_time' in info:
                    self.episode_coast_times.append(info['avg_coast_time'])
                if 'avg_nozzle_angle' in info:
                    self.episode_nozzle_angles.append(info['avg_nozzle_angle'])
                
                # Velocity statistics
                if 'avg_velocity' in info:
                    self.episode_velocities.append(info['avg_velocity'])
                
                # Reward components
                if 'avg_r_track' in info:
                    self.episode_r_tracks.append(info['avg_r_track'])
                if 'avg_r_heading' in info:
                    self.episode_r_headings.append(info['avg_r_heading'])
                if 'avg_r_cycle' in info:
                    self.episode_r_cycles.append(info['avg_r_cycle'])
                if 'avg_r_energy' in info:
                    self.episode_r_energies.append(info['avg_r_energy'])
                if 'avg_r_smooth' in info:
                    self.episode_r_smooths.append(info['avg_r_smooth'])
        
        # Log aggregated metrics periodically
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_aggregated_metrics()
        
        return True
    
    def _log_aggregated_metrics(self):
        """Log aggregated metrics from recent episodes."""
        
        # Navigation metrics
        if len(self.episode_successes) > 0:
            self.logger.record("custom/navigation/success_rate", np.mean(self.episode_successes))
            self.logger.record("custom/navigation/truncation_rate", np.mean(self.episode_truncations))
        
        if len(self.episode_final_distances) > 0:
            self.logger.record("custom/navigation/avg_final_distance", np.mean(self.episode_final_distances))
        
        # Path efficiency metrics
        if len(self.episode_path_lengths) > 0:
            self.logger.record("custom/path/path_length", np.mean(self.episode_path_lengths))
        
        if len(self.episode_direct_distances) > 0:
            self.logger.record("custom/path/direct_distance", np.mean(self.episode_direct_distances))
        
        if len(self.episode_efficiencies) > 0:
            self.logger.record("custom/path/efficiency", np.mean(self.episode_efficiencies))
        
        if len(self.episode_initial_distances) > 0:
            self.logger.record("custom/path/target_distance", np.mean(self.episode_initial_distances))
        
        # Performance metrics
        if len(self.episode_lengths) > 0:
            self.logger.record("custom/performance/avg_cycles", np.mean(self.episode_lengths))
        
        if len(self.episode_path_lengths) > 0 and len(self.episode_lengths) > 0:
            distances_per_cycle = [p/l for p, l in zip(self.episode_path_lengths, self.episode_lengths) if l > 0]
            if distances_per_cycle:
                self.logger.record("custom/performance/distance_per_cycle", np.mean(distances_per_cycle))
        
        # Only log cycles_to_success for successful episodes
        successful_lengths = [l for l, s in zip(self.episode_lengths, self.episode_successes) if s == 1.0]
        if successful_lengths:
            self.logger.record("custom/performance/cycles_to_success", np.mean(successful_lengths))
        
        # Action statistics
        if len(self.episode_compressions) > 0:
            self.logger.record("custom/actions/avg_compression", np.mean(self.episode_compressions))
            self.logger.record("custom/actions/compression_std", np.std(self.episode_compressions))
        
        if len(self.episode_coast_times) > 0:
            self.logger.record("custom/actions/avg_coast_time", np.mean(self.episode_coast_times))
        
        if len(self.episode_nozzle_angles) > 0:
            self.logger.record("custom/actions/avg_nozzle_angle", np.mean(self.episode_nozzle_angles))
            self.logger.record("custom/actions/nozzle_angle_std", np.std(self.episode_nozzle_angles))
        
        # Motion metrics
        if len(self.episode_velocities) > 0:
            self.logger.record("custom/motion/avg_velocity", np.mean(self.episode_velocities))
        
        # Reward components
        if len(self.episode_r_tracks) > 0:
            self.logger.record("reward/components/r_track", np.mean(self.episode_r_tracks))
        
        if len(self.episode_r_headings) > 0:
            self.logger.record("reward/components/r_heading", np.mean(self.episode_r_headings))
        
        if len(self.episode_r_cycles) > 0:
            self.logger.record("reward/components/r_cycle", np.mean(self.episode_r_cycles))
        
        if len(self.episode_r_energies) > 0:
            self.logger.record("reward/components/r_energy", np.mean(self.episode_r_energies))
        
        if len(self.episode_r_smooths) > 0:
            self.logger.record("reward/components/r_smooth", np.mean(self.episode_r_smooths))
        
        # Reward statistics
        if len(self.episode_rewards) > 0:
            self.logger.record("reward/stats/total_mean", np.mean(self.episode_rewards))
            self.logger.record("reward/stats/total_std", np.std(self.episode_rewards))
            self.logger.record("reward/stats/best_episode_reward", np.max(self.episode_rewards))
