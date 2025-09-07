# SALP Research Reference
## Bio-Inspired Soft Underwater Robot Research

*Research tracking for implementing SALP-inspired agents in gymnasium environment*

---

## Project Overview

**SALP** (Salp-inspired Approach to Low-energy Propulsion) - Bio-inspired soft underwater robot that swims via jet propulsion, mimicking marine invertebrate salps.

**Source**: University of Pennsylvania, Sung Robotics Lab  
**Research URL**: https://sung.seas.upenn.edu/research/bio-inspired-soft-underwater-robot-that-swims-via-jet-propulsion/

---

## Target Behavior (What We Want to Implement)

### Core Locomotion Mechanism
- **Jet Propulsion**: Volume-based water displacement for thrust
- **Expansion Cycle**: Contract → suck water in → expand → expel water → thrust forward
- **Pulsing Motion**: Rhythmic expansion/contraction like jellyfish
- **Natural Spring-Back**: Passive return to expanded state after contraction

### Visual Morphing
- **Body Shape**: Smooth barrel/ellipsoid (not origami)
- **State Changes**: 
  - **Contracted**: More spherical, smaller volume
  - **Expanded**: Ellipsoidal, larger volume
- **Smooth Transitions**: Gradual morphing between states

---

## Key Research Insights

### Biological Inspiration
- **Salps**: Barrel-shaped marine invertebrates
- **Natural Behavior**: Rapid body cavity volume changes
- **Water Flow**: Front aperture intake → rear funnel expulsion
- **Multi-Agent**: Can form "salp chains" for coordinated swimming

### Performance Characteristics
- **Swimming Speed**: ~6.7 cm/s (0.2 body lengths/s)
- **Multi-Robot Benefits**: 9% velocity increase, 16% acceleration boost when coordinated
- **Energy Efficiency**: Cost of transport ~2.0
- **Bidirectional**: Forward and reverse propulsion possible

---

## Implementation Focus

### Essential Mechanics for Simulation
1. **Volume-Based Physics**
   - Body size changes drive water displacement
   - Larger volume change = more thrust
   - Simple expansion/contraction timing

2. **Propulsion Cycle**
   - **Phase 1**: Contract body (motor + tendon simulation)
   - **Phase 2**: Natural spring-back expansion
   - **Result**: Directional water jet → forward thrust

3. **Visual Representation**
   - Smooth ellipsoid body (no complex geometry)
   - Dynamic size scaling during cycles
   - Optional: translucent/transparent appearance

### Multi-Agent Possibilities
- **Chain Formation**: Physical connections between agents
- **Coordinated Pulsing**: Synchronized or asynchronous cycles
- **Collective Benefits**: Improved swimming performance

---

## Research Evolution Timeline

- **2021**: Original origami-inspired robot (magic ball pattern)
- **2023**: Drag coefficient characterization (0.64-1.26 range)
- **2024-2025**: Multi-robot coordination, bidirectional control, self-sensing

---

## Key Publications

1. **"Origami-inspired robot that swims via jet propulsion"** (2021)
   - IEEE Robotics and Automation Letters
   - Original concept and implementation

2. **"Effect of Jet Coordination on Underwater Propulsion with the Multi-Robot SALP System"** (2025)
   - Multi-robot coordination benefits
   - Performance improvements quantified

3. **"Drag coefficient characterization of the origami magic ball"** (2023)
   - Fluid dynamics analysis
   - Shape-dependent drag characteristics

---

## Implementation Notes

### Current Gymnasium Environment
- **Implementation**: `salp_robot.py` - SALP-inspired robot with realistic breathing cycles
- **Features**: Volume-based morphing, steerable nozzle, realistic underwater physics
- **Key Components**: Body size changes during propulsion cycles, hold-to-inhale controls

### Simulation Features
- Morphing ellipsoid body with realistic breathing cycles
- Expansion/contraction animation with proper timing
- Volume-based thrust calculation
- Enhanced propulsion moment physics
- Optional: Multi-agent chain formation capability

### Physics Considerations
- **Thrust Calculation**: Based on volume change rate
- **Body Dynamics**: Smooth interpolation between contracted/expanded states
- **Water Interaction**: Volume displacement drives propulsion force

---

## Video References
- **Multi-Robot Demo**: https://www.youtube.com/watch?v=mzd1QCXssCk
- **Research Channel**: Sung Robotics Group (YouTube)

---

## Next Steps for Implementation

1. **Modify Current Simulation**: Enhance `squid_robot.py` with morphing body
2. **Add Volume Physics**: Implement expansion/contraction thrust mechanics
3. **Visual Enhancement**: Smooth body size transitions
4. **Multi-Agent**: Explore chain formation capabilities
5. **RL Training**: Develop agents that learn optimal pulsing patterns

---

## Additional Research - Dubins Vehicle Dynamics

### Key Papers Added:
1. **"Closed-Form Solutions for Minimum-Time Paths of Dubins Airplane in Steady Wind"** (2024)
   - ArXiv: https://arxiv.org/abs/2412.04797
   - Dubins vehicle path planning with wind effects
   - CSC, CCC, SC, CC path configurations

2. **"3D Dubins Curve-Based Path Planning for UUV in Unknown Environments Using an Improved RRT* Algorithm"** (2025)
   - MDPI: https://www.mdpi.com/2077-1312/13/7/1354
   - 3D Dubins curves for underwater vehicles
   - Nonholonomic constraints for UUVs

### Dubins Vehicle Principles:
- **Constant Forward Speed**: Vehicle always moves forward at fixed velocity
- **Minimum Turning Radius**: Cannot turn sharper than r_min constraint
- **Path Segments**: Combinations of straight lines (S) and circular arcs (C)
- **Optimal Paths**: CSC (curve-straight-curve), CCC (curve-curve-curve), etc.
- **No Lateral Movement**: Cannot move sideways (nonholonomic constraint)

### Critical Implementation Corrections:

#### **Breathing Cycle Timing**:
- **Previous**: 12 frames (0.2 seconds) - WAY TOO FAST
- **Corrected**: 2-3 seconds per phase (120-180 frames at 60fps)
- **Biological Reality**: Salps have slow, deliberate breathing cycles underwater

#### **Control Scheme**:
- **Previous**: Binary pulse trigger (instant on/off)
- **Corrected**: Hold-to-inhale system
  - **Hold SPACE**: Slow contraction (inhale water)
  - **Release SPACE**: Slow expansion (exhale water → thrust)
- **Underwater Physics**: Everything slower due to water resistance

#### **Movement Dynamics**:
- **Add Dubins Constraints**: Minimum turning radius for realistic movement
- **Body Orientation**: Agent points in movement direction
- **Smooth Turns**: No sharp direction changes, only curved paths
- **Forward Thrust**: Always in direction of body orientation

### Implementation Priority:
1. **Slow breathing cycles** (2-3 second phases)
2. **Hold-to-inhale controls** (space bar mechanics)
3. **Dubins vehicle dynamics** (turning radius constraints)
4. **Smooth body orientation** (gradual direction changes)

---

## Reinforcement Learning Research - SAC and Advanced Methods

### Soft Actor-Critic (SAC) Algorithm

**Source**: OpenAI Spinning Up Documentation  
**URL**: https://spinningup.openai.com/en/latest/algorithms/sac.html

#### Core SAC Principles:
- **Off-policy algorithm** for continuous action spaces
- **Entropy regularization**: Maximizes trade-off between expected return and entropy
- **Exploration-exploitation balance**: Higher entropy = more exploration
- **Clipped double-Q trick**: Uses minimum of two Q-networks for stability
- **Stochastic policy**: Inherent noise provides target policy smoothing

#### Key Technical Details:
- **Policy**: Squashed Gaussian with tanh activation for bounded actions
- **Networks**: Actor (policy), two Q-networks, target networks with polyak averaging
- **Reparameterization trick**: Enables gradient flow through stochastic actions
- **Temperature coefficient α**: Controls exploration vs exploitation trade-off

#### SAC vs Other Methods:
- **vs TD3**: SAC uses stochastic policy, TD3 uses deterministic + noise
- **vs DDPG**: SAC includes entropy regularization and double-Q
- **Advantages**: Better exploration, more stable training, handles stochasticity naturally

#### Implementation Notes for SALP:
- **Continuous control**: Perfect for SALP's continuous propulsion dynamics
- **Exploration**: Entropy bonus encourages diverse breathing patterns
- **Stability**: Double-Q prevents overestimation in complex underwater physics
- **Sample efficiency**: Off-policy learning from replay buffer

---

### CTSAC: Curriculum-Based Transformer SAC for Robot Exploration

**Source**: ArXiv paper (2025)  
**URL**: https://arxiv.org/html/2503.14254v1  
**Title**: "CTSAC: Curriculum-Based Transformer Soft Actor-Critic for Goal-Oriented Robot Exploration"

#### Key Innovations:
1. **Transformer-Enhanced SAC**: Integrates Transformer architecture into SAC for historical state reasoning
2. **Periodic Review Curriculum Learning**: Prevents catastrophic forgetting during training progression
3. **LiDAR Clustering Optimization**: Direction-aware segmentation for better forward perception
4. **Sim-to-Real Transfer**: ROS-Gazebo-PyTorch continuous training platform

#### Technical Architecture:
- **Actor Networks**: Actor Select (decision-making) + Actor Improve (policy updates)
- **Transformer Integration**: Processes T-step sequences for long-term reasoning
- **Attention Mechanism**: Captures correlations between different time steps
- **Curriculum Stages**: 6 progressive difficulty levels with periodic review

#### Relevance to SALP Implementation:
- **Historical Information**: Transformer can learn optimal breathing cycle patterns over time
- **Curriculum Learning**: Progressive training from simple to complex underwater environments
- **Continuous Control**: Proven effectiveness in continuous robotic control tasks
- **Exploration Strategy**: Advanced exploration techniques for complex environments

#### Performance Results:
- **Success Rate**: 80% in real-world experiments
- **Training Time**: ~15.2 hours (vs 8.5 days for standard SAC)
- **Generalization**: Superior performance across diverse test environments
- **Sim-to-Real**: Successful transfer to real robot platforms

#### Implementation Insights for SALP:
1. **Sequence Learning**: Use Transformer to learn optimal propulsion sequences
2. **Curriculum Design**: Start with simple swimming, progress to complex navigation
3. **Historical Context**: Leverage past breathing cycles for better decision-making
4. **Multi-Agent**: Framework supports coordinated multi-SALP exploration

---

---

## Papers to Access Later (Content Not Retrieved)

### 🔴 Priority Papers - Need Manual Access

1. **"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"** (2018)
   - **Authors**: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
   - **URL**: https://arxiv.org/pdf/1801.01290
   - **Status**: ❌ PDF fetch failed - need manual download
   - **Importance**: ⭐⭐⭐ Original foundational SAC paper
   - **Key Topics**: Entropy-regularized RL, maximum entropy framework, stochastic policies
   - **Relevance to SALP**: Core algorithm understanding for implementation

2. **"Soft Actor-Critic Algorithms and Applications"** (2018)
   - **Authors**: Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine
   - **URL**: https://arxiv.org/pdf/1812.05905
   - **Status**: ❌ PDF fetch failed - need manual download
   - **Importance**: ⭐⭐⭐ Practical SAC applications and improvements
   - **Key Topics**: SAC variants, real-world applications, algorithmic improvements
   - **Relevance to SALP**: Implementation best practices and real-world deployment

3. **"Soft Actor-Critic Deep Reinforcement Learning for Robotics"** (Finn Rietz)
   - **Author**: Finn Rietz
   - **URL**: https://www.finnrietz.dev/assets/img/rl_sac_analysis/Finn_Rietz_Soft_Actor_Critic_Deep_Reinforcement_Learning_for_Robotics_Paper.pdf
   - **Status**: ❌ PDF fetch failed - need manual download
   - **Importance**: ⭐⭐ Robotics-specific SAC analysis
   - **Key Topics**: SAC in robotics, practical considerations, implementation analysis
   - **Relevance to SALP**: Direct robotics applications and lessons learned

4. **"Reinforcement Learning for Autonomous Underwater Vehicle Control"** (2023)
   - **Authors**: Unknown (need to access paper for details)
   - **URL**: https://arxiv.org/pdf/2304.00225
   - **Status**: ❌ PDF fetch failed - need manual download
   - **Importance**: ⭐⭐ Recent RL work for underwater vehicles
   - **Key Topics**: Autonomous underwater vehicle control, reinforcement learning applications
   - **Relevance to SALP**: Direct underwater robotics RL applications

5. **ArXiv Paper 2506.15082v1** (2025)
   - **Authors**: Unknown (need to access paper for details)
   - **URL**: https://arxiv.org/html/2506.15082v1#S1
   - **Status**: ❌ HTML fetch timeout - need manual access
   - **Importance**: ⭐⭐ Recent research (2025)
   - **Key Topics**: Unknown (need to access paper for details)
   - **Relevance to SALP**: To be determined after accessing paper content

### 📋 Action Items for These Papers:
- [ ] Download PDFs manually from provided URLs
- [ ] Extract key insights relevant to SALP underwater robotics
- [ ] Add detailed summaries to research collection
- [ ] Identify specific implementation techniques for underwater environments
- [ ] Compare approaches with successfully analyzed CTSAC paper

### 🔍 Alternative Access Methods:
- Try accessing through institutional library systems
- Check for open access versions on author websites
- Look for related conference presentations or slides
- Search for implementation code repositories

---

### Successfully Analyzed Papers:

#### ✅ OpenAI Spinning Up SAC Documentation
- **URL**: https://spinningup.openai.com/en/latest/algorithms/sac.html
- **Status**: ✅ Successfully fetched and analyzed
- **Content**: Comprehensive SAC algorithm documentation with implementation details

#### ✅ CTSAC: Curriculum-Based Transformer SAC (2025)
- **URL**: https://arxiv.org/html/2503.14254v1
- **Status**: ✅ Successfully fetched and analyzed
- **Content**: Advanced SAC variant with Transformer integration and curriculum learning

#### ✅ End-to-End AUV Motion Planning Method Based on SAC (2021)
- **URL**: https://pubmed.ncbi.nlm.nih.gov/34502781/
- **Status**: ✅ Successfully fetched and analyzed
- **Content**: SAC + GAIL for underwater vehicle motion planning with comprehensive reward design

---

### End-to-End AUV Motion Planning Method Based on SAC

**Source**: Sensors Journal (2021)  
**URL**: https://pubmed.ncbi.nlm.nih.gov/34502781/  
**Title**: "End-to-End AUV Motion Planning Method Based on Soft Actor-Critic"

#### Key Contributions:
1. **SAC + GAIL Integration**: Combines Soft Actor-Critic with Generative Adversarial Imitation Learning
2. **End-to-End System**: Direct mapping from AUV/environment state to control instructions
3. **Comprehensive Reward Function**: Multi-component reward design for smooth navigation
4. **Unity Simulation Platform**: 3D underwater environment for training and testing

#### Technical Implementation:
- **Algorithm**: SAC enhanced with GAIL for faster training
- **Environment**: Unity-based 3D underwater simulation
- **AUV Model**: Underactuated autonomous underwater vehicle
- **Sensors**: Sonar-based perception system
- **Control**: Surge force and yaw moment outputs

#### Reward Function Design (Highly Relevant to SALP):
1. **Goal Reaching**: Positive reward for reaching target
2. **Collision Avoidance**: Negative penalty for obstacles
3. **Distance Optimization**: Reward for moving closer to goal
4. **Time Efficiency**: Penalty for excessive time/steps
5. **Smooth Navigation**: Penalties for erratic movements
6. **Energy Efficiency**: Considerations for control effort

#### Performance Results:
- **Navigation Quality**: Optimal decision-making during navigation
- **Path Efficiency**: Shorter routes compared to traditional methods
- **Time Performance**: Reduced time consumption
- **Trajectory Smoothness**: Smoother paths than baseline algorithms
- **Training Acceleration**: GAIL significantly reduced training time

#### Direct Applications to SALP:
1. **Reward Structure**: Comprehensive reward design applicable to SALP breathing cycles
2. **End-to-End Learning**: Direct state-to-action mapping for propulsion control
3. **GAIL Integration**: Could accelerate SALP training with expert demonstrations
4. **Underwater Physics**: Proven effectiveness in underwater environments
5. **Multi-Objective Optimization**: Balance between efficiency, smoothness, and goal achievement

#### Key Insights for SALP Implementation:
- **State Representation**: Include AUV position, velocity, goal direction, obstacle distances
- **Action Space**: Continuous control for propulsion forces and steering
- **Training Strategy**: Use GAIL with expert demonstrations to bootstrap learning
- **Reward Engineering**: Multi-component rewards for complex underwater behaviors
- **Simulation Platform**: Unity provides realistic 3D underwater physics

---

### Implementation Recommendations for SALP:

#### SAC Configuration for Underwater Robotics:
- **Action Space**: Continuous control for breathing cycle timing and intensity
- **State Space**: Include historical breathing states, water flow, position, velocity
- **Reward Design**: 
  - Positive: Forward progress, efficient propulsion, goal reaching
  - Negative: Energy waste, collision, stagnation
  - Entropy bonus: Encourage diverse exploration strategies

#### Advanced Features to Consider:
1. **Transformer Integration**: For learning complex breathing patterns over time
2. **Curriculum Learning**: Progressive difficulty in swimming environments
3. **Multi-Agent Coordination**: For SALP chain formation and collective swimming
4. **Sim-to-Real Transfer**: Robust training for real underwater deployment

#### Training Strategy:
1. **Start Simple**: Basic propulsion in open water
2. **Add Complexity**: Obstacles, currents, navigation goals
3. **Multi-Agent**: Coordinated swimming and chain formation
4. **Real Transfer**: Deploy to actual underwater environments

---

*Last Updated: January 2025*
*Focus: Simplified implementation for gymnasium environment experimentation*
*Updated: Added Dubins vehicle dynamics, corrected breathing cycle timing, and comprehensive SAC/RL research*
