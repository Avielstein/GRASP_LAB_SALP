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

*Last Updated: January 2025*
*Focus: Simplified implementation for gymnasium environment experimentation*
*Updated: Added Dubins vehicle dynamics and corrected breathing cycle timing*
