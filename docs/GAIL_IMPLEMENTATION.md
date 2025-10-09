# GAIL Implementation for SALP Robot

Complete documentation for GAIL (Generative Adversarial Imitation Learning) integration with SAC.

## Overview

This implementation combines Soft Actor-Critic (SAC) with GAIL to enable learning from expert demonstrations. The hybrid approach allows the agent to benefit from both hand-crafted environment rewards and learned rewards from expert behavior.

## Architecture

### Components

1. **Discriminator Network** (`agents/discriminator.py`)
   - Binary classifier for (state, action) pairs
   - Distinguishes expert demonstrations from agent behavior
   - Provides GAIL reward signal: `-log(1 - D(s,a))`

2. **SAC-GAIL Agent** (`agents/sac_gail_agent.py`)
   - Extends standard SAC agent
   - Integrates discriminator training
   - Supports hybrid reward combining environment + GAIL

3. **Expert Buffer** (`training/expert_buffer.py`)
   - Stores and manages expert demonstrations
   - Supports both human and agent demonstrations
   - Provides sampling interface for discriminator training

4. **Collection Scripts**
   - `scripts/collect_human_demos.py` - Manual gameplay recording
   - `scripts/collect_agent_demos.py` - Automated collection from trained agents

## Quick Start

### Step 1: Collect Human Demonstrations

```bash
# Record 5-10 episodes of your best gameplay
python scripts/collect_human_demos.py
```

Play the environment and save successful episodes. This provides initial expert guidance.

### Step 2: (Optional) Collect Agent Demonstrations

If you have a trained SAC model:

```bash
# Extract top-performing episodes from trained agent
python scripts/collect_agent_demos.py \
    --model models/your_model/best_model.pth \
    --num-episodes 50 \
    --top-k 20 \
    --min-score 100.0
```

### Step 3: Train with GAIL

```bash
# Train SAC+GAIL agent using collected demonstrations
python scripts/train_sac_gail.py
```

## Configuration

### GAIL Parameters (config/base_config.py)

```python
@dataclass
class GAILConfig:
    use_gail: bool = False                    # Enable/disable GAIL
    expert_demos_path: str = "expert_demos/"  # Path to demonstrations
    discriminator_lr: float = 3e-4            # Discriminator learning rate
    discriminator_update_freq: int = 1        # Update frequency
    reward_env_weight: float = 0.3            # Environment reward weight
    reward_gail_weight: float = 0.7           # GAIL reward weight
    min_expert_episodes: int = 5              # Minimum demos required
    load_human_demos: bool = True             # Load human demonstrations
    load_agent_demos: bool = True             # Load agent demonstrations
```

### Reward Weighting Strategy

The total reward is computed as:
```
total_reward = λ_env * env_reward + λ_gail * gail_reward
```

**Recommended schedules:**
- **Early training**: High GAIL weight (0.7) for expert guidance
- **Late training**: Increase env weight for task optimization
- **Pure imitation**: Set env_weight = 0, gail_weight = 1.0

## How GAIL Works

### Training Loop

1. **Agent acts** in environment → generates (state, action) pairs
2. **Discriminator classifies** expert vs agent pairs
3. **GAIL reward** computed from discriminator output
4. **SAC updates** using hybrid reward
5. **Discriminator updates** to maintain discrimination

### Discriminator Loss

```python
# Expert samples should be classified as 1 (from expert)
expert_loss = BCE(D(s_expert, a_expert), 1)

# Agent samples should be classified as 0 (not from expert)
agent_loss = BCE(D(s_agent, a_agent), 0)

total_loss = expert_loss + agent_loss
```

### GAIL Reward Function

```python
# Encourages agent to fool discriminator
gail_reward = -log(1 - D(s, a) + ε)
```

High reward when discriminator thinks agent's action is expert-like.

## Usage Examples

### Basic Training

```bash
# Train with default configuration
python scripts/train_sac_gail.py
```

### Custom Reward Weights

```bash
# More emphasis on environment reward
python scripts/train_sac_gail.py --env-weight 0.5 --gail-weight 0.5

# Pure imitation learning
python scripts/train_sac_gail.py --env-weight 0.0 --gail-weight 1.0
```

### Only Human Demonstrations

```bash
# Train using only human demonstrations
python scripts/train_sac_gail.py --no-agent-demos
```

### Custom Expert Path

```bash
# Use demonstrations from different directory
python scripts/train_sac_gail.py --expert-path path/to/demos/
```

## Performance Benefits

Based on research (AUV paper with SAC+GAIL):

- **15x faster training**: 15 hours vs 8 days for pure SAC
- **Better sample efficiency**: Learns from expert experience
- **Improved exploration**: Expert guidance reduces random exploration
- **Smoother trajectories**: Mimics expert movement patterns

## Best Practices

### Demonstration Collection

1. **Quality matters**: 10 good demos > 100 mediocre ones
2. **Diversity**: Show different successful strategies
3. **Mix sources**: Combine human intuition + agent performance
4. **Minimum threshold**: Aim for 5-10 demonstrations minimum
5. **High performance**: Only save episodes with good scores

### Training Strategy

1. **Start with human demos**: Bootstrap learning with intuitive behavior
2. **Add agent demos**: Scale up with automated high-performance examples
3. **Monitor discriminator**: Accuracy ~50-70% indicates good balance
4. **Adjust weights**: Fine-tune reward balance based on performance
5. **Curriculum learning**: Gradually shift from GAIL to environment reward

### Troubleshooting

**Discriminator too accurate (>90%)**
- Agent not learning from demos effectively
- Increase GAIL reward weight
- Add more diverse demonstrations

**Discriminator too weak (<50%)**
- Agent ignoring expert guidance
- Check expert buffer is loaded correctly
- Increase discriminator learning rate

**No improvement over SAC**
- Expert demonstrations may be poor quality
- Try collecting better demos
- Verify hybrid reward is being used

## File Structure

```
agents/
├── discriminator.py           # Discriminator network
├── sac_agent.py              # Base SAC agent
└── sac_gail_agent.py         # SAC + GAIL integration

training/
├── expert_buffer.py          # Expert demonstration management
└── trainer.py                # Training loop with GAIL support

scripts/
├── collect_human_demos.py    # Manual demonstration collection
├── collect_agent_demos.py    # Automated demo collection
└── train_sac_gail.py         # GAIL training script

expert_demos/
├── human/                    # Human demonstrations
├── agent/                    # Agent demonstrations
└── README.md                 # Collection guide

config/
└── base_config.py            # GAIL configuration
```

## Algorithm Details

### SAC+GAIL Update Procedure

```python
for episode in training:
    for step in episode:
        # 1. Select action with SAC policy
        action = policy.sample(state)
        
        # 2. Execute in environment
        next_state, env_reward, done = env.step(action)
        
        # 3. Compute GAIL reward
        gail_reward = -log(1 - discriminator(state, action))
        
        # 4. Hybrid reward
        total_reward = λ_env * env_reward + λ_gail * gail_reward
        
        # 5. Store in replay buffer
        buffer.add(state, action, env_reward, next_state, done)
        
        # 6. Update SAC (Q, policy, alpha)
        if step % update_freq == 0:
            batch = buffer.sample()
            sac_agent.update(batch)
        
        # 7. Update discriminator
        if step % disc_update_freq == 0:
            expert_batch = expert_buffer.sample()
            agent_batch = buffer.sample()
            discriminator.update(expert_batch, agent_batch)
```

## Research Background

This implementation is based on:

1. **GAIL Paper**: "Generative Adversarial Imitation Learning" (Ho & Ermon, 2016)
2. **SAC Paper**: "Soft Actor-Critic" (Haarnoja et al., 2018)
3. **AUV Application**: "End-to-End AUV Motion Planning Method Based on SAC" (2021)
   - Demonstrated 15x training speedup with GAIL
   - Effective for underwater robotics
   - Hybrid reward approach proven successful

## Advanced Features

### Dynamic Reward Scheduling

Adjust reward weights during training:

```python
# In your training script
if episode < 500:
    agent.set_reward_weights(env_weight=0.2, gail_weight=0.8)
elif episode < 1000:
    agent.set_reward_weights(env_weight=0.5, gail_weight=0.5)
else:
    agent.set_reward_weights(env_weight=0.8, gail_weight=0.2)
```

### Discriminator Accuracy Monitoring

```python
# Check discriminator performance
accuracy = agent.get_discriminator_accuracy(batch)
print(f"Discriminator accuracy: {accuracy:.2%}")
```

### Expert Buffer Filtering

```python
# Filter expert buffer by minimum reward
expert_buffer.filter_by_reward(min_reward=50.0)
```

## Comparison: SAC vs SAC+GAIL

| Aspect | SAC | SAC+GAIL |
|--------|-----|----------|
| Training Time | Longer | **15x faster** |
| Sample Efficiency | Lower | **Higher** |
| Expert Knowledge | Not used | **Leveraged** |
| Exploration | Random | **Guided** |
| Initial Performance | Slow | **Fast bootstrap** |
| Final Performance | Good | **Better** |
| Setup Complexity | Simple | Moderate |

## FAQ

**Q: How many demonstrations do I need?**
A: Start with 5-10, but more is better. Quality > quantity.

**Q: Should I use human or agent demonstrations?**
A: Both! Human demos for intuition, agent demos for scale.

**Q: Can I train without any demonstrations?**
A: Yes, set `use_gail=False` in config for standard SAC.

**Q: Why is GAIL faster than pure SAC?**
A: Expert guidance reduces random exploration and provides better initial policy.

**Q: What if discriminator accuracy is 100%?**
A: Increase GAIL reward weight or add more diverse demonstrations.

## Next Steps

1. ✅ Collect expert demonstrations
2. ✅ Verify demonstrations with expert buffer
3. ✅ Train SAC+GAIL agent
4. ✅ Monitor discriminator accuracy
5. ✅ Compare performance with baseline SAC
6. ✅ Adjust reward weights if needed

## Support

For issues or questions:
- Check `expert_demos/README.md` for demonstration guidelines
- Review research papers in `docs/SALP_RESEARCH.md`
- Verify configuration in `config/base_config.py`

---

**Implementation Status**: ✅ Complete and ready for use

**Last Updated**: January 2025
