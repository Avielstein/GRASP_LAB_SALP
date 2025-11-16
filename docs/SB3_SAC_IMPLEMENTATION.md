# Stable Baselines3 SAC Implementation

This document describes the integration of Stable Baselines3's SAC implementation into the SALP project.

## Overview

The project now supports two SAC implementations:

1. **Custom SAC** (`agents/sac_agent.py`) - Original custom PyTorch implementation
2. **Stable Baselines3 SAC** (`agents/sb3_sac_agent.py`) - Standard library implementation ✨ **NEW**

## Why Use Stable Baselines3?

### Advantages
- ✅ **Well-tested and maintained** - Industry-standard library with extensive testing
- ✅ **Easy to compare** - Standard implementation matches published papers
- ✅ **Better documentation** - Extensive docs and examples
- ✅ **Built-in features** - Callbacks, logging, evaluation, checkpointing
- ✅ **Community support** - Large user base and active development
- ✅ **Reproducibility** - Easier to reproduce results and compare with other work

### When to Use Each Implementation

**Use SB3 SAC when:**
- Starting new experiments
- Benchmarking performance
- Need reliable, tested code
- Want standard hyperparameters
- Comparing with other research

**Use Custom SAC when:**
- Need specific modifications to the algorithm
- Integrating with GAIL (SAC-GAIL agent)
- Research into algorithm variations
- Already have trained models with custom implementation

## Installation

```bash
pip install stable-baselines3[extra]>=2.0.0
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with SB3 SAC (recommended for baseline)
python scripts/train_sb3_sac.py

# Or with custom SAC
python scripts/train_single_food_optimal.py
```

### Testing

```bash
# Test a trained SB3 model
python scripts/test_sb3_sac.py models/sb3_sac_20241116_141234/best_model.zip --episodes 10

# Test with rendering
python scripts/test_sb3_sac.py models/sb3_sac_20241116_141234/best_model.zip --render
```

## Usage Examples

### Basic Training

```python
from agents.sb3_sac_agent import SB3SACAgent
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config

# Setup
config = get_sac_snake_config()
env = SalpSnakeEnv(render_mode=None, **config.environment.params)
agent = SB3SACAgent(config.agent, env, verbose=1)

# Train
agent.learn(total_timesteps=100000)

# Save
agent.save("models/my_model")
```

### Loading and Testing

```python
from stable_baselines3 import SAC
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config

# Load
config = get_sac_snake_config()
env = SalpSnakeEnv(render_mode=None, **config.environment.params)
model = SAC.load("models/my_model", env=env)

# Test
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

### With Callbacks

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpoints/",
    name_prefix="sac_checkpoint"
)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    eval_freq=5000,
    n_eval_episodes=5
)

# Train with callbacks
agent.learn(
    total_timesteps=100000,
    callback=[checkpoint_callback, eval_callback]
)
```

## Configuration

The SB3 SAC agent uses the same configuration as the custom SAC agent from `config/base_config.py`:

```python
agent = AgentConfig(
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    replay_buffer_size=1000000,
    hidden_sizes=[256, 256],
    activation="relu"
)
```

Additional SB3-specific parameters can be set in `config.agent.params`:

```python
params = {
    'alpha': 'auto',              # Entropy coefficient (auto-tune)
    'learning_starts': 1000,      # Steps before learning
    'train_freq': 1,              # Training frequency
    'gradient_steps': 1,          # Gradient steps per update
    'target_update_interval': 1,  # Target network update frequency
    'seed': 42,                   # Random seed
    'device': 'auto',             # 'cpu', 'cuda', or 'auto'
}
```

## File Structure

```
GRASP_LAB_SALP/
├── agents/
│   ├── sac_agent.py           # Custom SAC implementation
│   ├── sb3_sac_agent.py       # SB3 SAC wrapper ✨ NEW
│   └── sac_gail_agent.py      # SAC+GAIL (uses custom SAC)
├── scripts/
│   ├── train_sb3_sac.py       # Training script for SB3 SAC ✨ NEW
│   ├── test_sb3_sac.py        # Testing script for SB3 SAC ✨ NEW
│   └── train_single_food_optimal.py  # Training with custom SAC
└── docs/
    └── SB3_SAC_IMPLEMENTATION.md  # This file ✨ NEW
```

## Comparison: Custom vs SB3 SAC

| Feature | Custom SAC | SB3 SAC |
|---------|-----------|----------|
| Implementation | Manual PyTorch | Stable Baselines3 |
| Testing | Limited | Extensively tested |
| Documentation | Internal | Comprehensive |
| Maintenance | Manual | Library maintained |
| GAIL Integration | ✅ Yes | ❌ No (use custom) |
| Callbacks | Manual | Built-in |
| Logging | Basic | TensorBoard ready |
| Checkpointing | Manual | Automatic |
| Reproducibility | Good | Excellent |

## Migration Guide

### From Custom SAC to SB3 SAC

If you have existing code using custom SAC, here's how to migrate:

**Before (Custom SAC):**
```python
from agents.sac_agent import SACAgent

agent = SACAgent(config.agent, obs_dim, action_dim, action_space)

# Manual training loop required
for step in range(max_steps):
    action = agent.select_action(obs)
    # ... collect experience ...
    metrics = agent.update(batch)
```

**After (SB3 SAC):**
```python
from agents.sb3_sac_agent import SB3SACAgent

agent = SB3SACAgent(config.agent, env, verbose=1)

# Built-in training loop
agent.learn(total_timesteps=max_steps)
```

## Performance Notes

- Both implementations should achieve similar performance on the SALP Snake environment
- SB3 SAC may be slightly more stable due to extensive tuning and testing
- Training times are comparable
- SB3 includes more optimizations for replay buffer and gradient computation

## Troubleshooting

### Import Error
```
ImportError: No module named 'stable_baselines3'
```
**Solution:** Install stable-baselines3: `pip install stable-baselines3[extra]>=2.0.0`

### GPU Not Used
**Solution:** Check PyTorch CUDA installation. SB3 will automatically use GPU if available.

### Slow Training
**Solution:** 
- Increase `train_freq` and `gradient_steps` in config
- Ensure using GPU if available
- Check environment step time

## Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [SB3 SAC Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Training Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

## Next Steps

1. **Benchmark**: Train both implementations and compare performance
2. **Hyperparameter tuning**: Use SB3's built-in tools for optimization
3. **Integration**: Consider using SB3 as base for GAIL experiments
4. **Monitoring**: Set up TensorBoard logging for better visualization

## Contributing

When adding new features:
- Keep both implementations if feasible
- Prefer SB3 for standard RL experiments
- Document differences and use cases
- Add tests for new functionality
