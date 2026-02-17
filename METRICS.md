# TensorBoard Metrics Documentation

This document explains all metrics logged to TensorBoard during SALP robot training.

## Table of Contents
- [Standard SB3 Metrics](#standard-sb3-metrics)
- [Custom Navigation Metrics](#custom-navigation-metrics)
- [Reward Component Breakdown](#reward-component-breakdown)
- [Physical Performance Metrics](#physical-performance-metrics)
- [Diagnostic Metrics](#diagnostic-metrics)

---

## Standard SB3 Metrics

These are automatically logged by Stable Baselines3's SAC algorithm:

### Training Metrics (`train/`)
- **`train/actor_loss`**: Loss of the policy network (actor). Lower is better. Measures how well the policy is learning.
- **`train/critic_loss`**: Loss of the value networks (critics). Measures prediction accuracy of Q-values.
- **`train/ent_coef`**: Entropy coefficient. Controls exploration vs exploitation trade-off. Higher = more exploration.
- **`train/ent_coef_loss`**: Loss associated with entropy coefficient optimization.
- **`train/learning_rate`**: Current learning rate for gradient descent.
- **`train/n_updates`**: Total number of gradient descent updates performed.

### Rollout Metrics (`rollout/`)
- **`rollout/ep_len_mean`**: Average episode length in steps (breathing cycles). Shows how long episodes last.
- **`rollout/ep_rew_mean`**: **Average cumulative reward per episode**. Primary training objective. Higher is better.

### Time Metrics (`time/`)
- **`time/fps`**: Training speed in frames (steps) per second. Higher = faster training.
- **`time/iterations`**: Number of training iterations completed.
- **`time/time_elapsed`**: Wall clock time since training started (seconds).
- **`time/total_timesteps`**: Total environment steps executed across all episodes.

### Evaluation Metrics (`eval/`)
- **`eval/mean_reward`**: Average reward during evaluation episodes (deterministic policy).
- **`eval/mean_ep_length`**: Average episode length during evaluation.

---

## Custom Navigation Metrics

These track how well the robot navigates to targets:

### Success and Completion (`custom/navigation/`)
- **`custom/navigation/success_rate`**: Percentage of episodes where robot reaches target (distance < 0.05m). Range: [0, 1]. Target: > 0.8
- **`custom/navigation/truncation_rate`**: Percentage of episodes that hit max cycles or go out of bounds. Lower is better.
- **`custom/navigation/avg_final_distance`**: Average distance from target at episode end (meters). Lower is better. Target: < 0.1m

### Path Efficiency (`custom/path/`)
- **`custom/path/path_length`**: Total distance robot traveled (sum of all movements) in meters. Shows how far robot swims.
- **`custom/path/direct_distance`**: Straight-line distance from start to end position (meters).
- **`custom/path/efficiency`**: Ratio of direct distance to path length. Range: [0, 1]. Higher = straighter path. Target: > 0.7
  - Formula: `direct_distance / path_length`
  - 1.0 = perfectly straight line
  - 0.5 = robot traveled 2x the optimal distance
- **`custom/path/target_distance`**: Initial distance to target at episode start (meters). Used for normalization.

### Performance (`custom/performance/`)
- **`custom/performance/avg_cycles`**: Average number of breathing cycles per episode. Lower = more efficient.
- **`custom/performance/distance_per_cycle`**: Average distance traveled per breathing cycle (m/cycle). Higher = more efficient propulsion.
- **`custom/performance/time_per_episode`**: Average wall-clock time per episode (seconds).
- **`custom/performance/cycles_to_success`**: Average cycles needed to reach target (only for successful episodes).

---

## Reward Component Breakdown

The total reward is composed of 5 weighted components. These help diagnose what the agent is learning:

### Component Metrics (`reward/components/`)
- **`reward/r_track`**: Distance improvement reward. Positive when moving toward target, negative when moving away.
  - Formula: `(prev_dist - current_dist) * 100`
  - Typical range: [-5, +5] per cycle
  - Goal: Maximize by moving directly toward target

- **`reward/r_heading`**: Heading alignment reward. Measures if velocity vector points toward target.
  - Formula: `dot_product(velocity_direction, target_direction)`
  - Range: [-1, +1]
  - +1.0 = moving directly toward target
  - 0.0 = moving perpendicular
  - -1.0 = moving away from target

- **`reward/r_cycle`**: Time penalty for each breathing cycle. Encourages efficiency.
  - Value: -0.5 per cycle
  - Total penalty accumulates over episode
  - Goal: Reach target in fewer cycles

- **`reward/r_energy`**: Energy efficiency reward. Encourages strong compressions.
  - Formula: `-0.1 * (1 - compression)^2`
  - Range: [-0.1, 0.0]
  - Penalizes weak compressions (< 100% of max)
  - Goal: Use full compression for maximum thrust

- **`reward/r_smooth`**: Control smoothness reward. Penalizes erratic steering.
  - Formula: `-0.1 * (nozzle_angle_change)^2`
  - Range: [-0.1, 0.0]
  - Penalizes large changes in nozzle direction
  - Goal: Smooth, predictable steering

### Aggregated Reward Stats (`reward/stats/`)
- **`reward/total_mean`**: Mean total reward across recent episodes (same as `rollout/ep_rew_mean`).
- **`reward/total_std`**: Standard deviation of total reward. Lower = more consistent performance.
- **`reward/best_episode_reward`**: Highest reward achieved in recent episodes.

---

## Physical Performance Metrics

Track the robot's physical behavior:

### Action Statistics (`custom/actions/`)
- **`custom/actions/avg_compression`**: Average compression amount used (0-1 scale). 1.0 = maximum compression.
- **`custom/actions/avg_coast_time`**: Average coast time in seconds (0-10s). Shows how long robot coasts after jet.
- **`custom/actions/avg_nozzle_angle`**: Average absolute nozzle steering angle in radians. Shows typical steering magnitude.
- **`custom/actions/compression_std`**: Standard deviation of compression. Higher = more varied thrust levels.
- **`custom/actions/nozzle_angle_std`**: Standard deviation of nozzle angles. Higher = more varied steering.

### Motion Metrics (`custom/motion/`)
- **`custom/motion/avg_velocity`**: Average velocity magnitude (m/s). Shows typical swimming speed.
- **`custom/motion/max_velocity`**: Peak velocity achieved during episode (m/s).
- **`custom/motion/avg_angular_velocity`**: Average body rotation rate (rad/s). Shows turning behavior.
- **`custom/motion/trajectory_smoothness`**: Measure of path smoothness (1.0 = perfectly smooth, 0.0 = very jerky).

---

## Diagnostic Metrics

Help identify training issues:

### Problem Detection (`custom/diagnostics/`)
- **`custom/diagnostics/stall_rate`**: Percentage of steps where robot isn't making progress toward target. Lower is better.
  - Stall = distance to target not decreasing for multiple cycles
  
- **`custom/diagnostics/oscillation_rate`**: Percentage of episodes with back-and-forth motion. Lower is better.
  - Oscillation = repeatedly overshooting target or zigzagging

- **`custom/diagnostics/stuck_rate`**: Percentage of steps where robot is essentially motionless. Lower is better.
  - Stuck = velocity < 0.01 m/s for multiple cycles

- **`custom/diagnostics/avg_distance_to_target`**: Average distance to target across all steps (not just final). Lower is better.

---

## How to Use These Metrics

### During Training:
1. **Monitor `custom/navigation/success_rate`** - Is it increasing? Target: > 80%
2. **Check `custom/path/efficiency`** - Are paths getting straighter? Target: > 70%
3. **Watch `reward/r_track`** - Is agent learning to move toward target?
4. **Track `custom/performance/distance_per_cycle`** - Is propulsion improving?

### Debugging Poor Performance:
- **Low success rate + high path length** → Agent taking inefficient paths
- **High success rate + low efficiency** → Agent reaching targets but zigzagging
- **Positive r_track but low success** → Good direction but too slow
- **High stall_rate** → Agent getting stuck or confused
- **High oscillation_rate** → Agent overshooting and correcting repeatedly

### Comparing Models:
- Compare `custom/navigation/success_rate` across versions
- Compare `custom/path/efficiency` for navigation quality
- Compare `custom/performance/cycles_to_success` for efficiency
- Compare `reward/r_smooth` for control quality

---

## TensorBoard Commands

### View Logs:
```bash
tensorboard --logdir experiments/v3/logs/
```

### Compare Multiple Versions:
```bash
tensorboard --logdir experiments/ --port 6006
```

### Filter Specific Metrics:
In TensorBoard UI, use the search bar to filter by prefix:
- `custom/navigation/*` - Navigation metrics
- `reward/*` - Reward breakdown
- `custom/actions/*` - Action statistics

---

## Expected Metric Ranges for Well-Trained Agent

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Success Rate | < 50% | 50-70% | 70-85% | > 85% |
| Path Efficiency | < 50% | 50-65% | 65-75% | > 75% |
| Distance/Cycle | < 0.1m | 0.1-0.2m | 0.2-0.3m | > 0.3m |
| Avg Cycles | > 20 | 15-20 | 10-15 | < 10 |
| r_track (per cycle) | < 0 | 0-1 | 1-2 | > 2 |
| r_smooth | < -2 | -2 to -1 | -1 to -0.5 | > -0.5 |

---

## Version History

### v1 (Baseline)
- Basic reward: distance improvement only
- No cycle penalty
- Success rate: ~60-70%

### v2 (Enhanced Reward)
- Added: r_cycle, r_energy components
- Improved efficiency focus
- Success rate: ~70-80%

### v3 (Current - With Detailed Logging)
- All metrics documented above
- Full observability into learning process
- Target success rate: > 85%

---

Last updated: 2026-02-16
