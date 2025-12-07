# Navigation Evaluation Tools

Simple tools to evaluate SALP robot navigation performance.

## Tools

### 1. Watch Navigation (Live Viewing)
Watch a single navigation trial in real-time.

```bash
python eval/watch_navigation.py \
  --model data/models/YOUR_MODEL.zip \
  --start 150,300 \
  --goal 650,300
```

### 2. Batch Evaluation (Statistics & Visualizations)
Run multiple trials and generate statistics + visualizations.

```bash
# Generate trajectory plot
python eval/navigation_heatmap.py \
  --model data/models/YOUR_MODEL.zip \
  --trials 30 \
  --start 150,300 \
  --goal 650,300 \
  --plot-trajectories

# Generate heatmap (omit --plot-trajectories)
python eval/navigation_heatmap.py \
  --model data/models/YOUR_MODEL.zip \
  --trials 30 \
  --start 150,300 \
  --goal 650,300
```

## Metrics

- **Path Efficiency**: Ratio of actual path length to optimal (straight-line) distance
- **Success Rate**: Percentage of trials reaching goal (within 50px)
- **Area Coverage**: 2D area explored (bounding box)
- **Straightness**: How direct the path is

## Results

Outputs saved to `eval/results/`:
- Trajectory plots: `navigation_trajectories_*.png`
- Heatmaps: `navigation_heatmap_*.png`
- Statistics: `navigation_stats_*.json`
