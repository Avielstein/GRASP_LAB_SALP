# test_robot.py
import numpy as np
from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def generate_circle_trajectory(center, radius, num_points=20):
    """
    Generate a circular trajectory.
    
    Args:
        center: Center point [x, y] in meters
        radius: Radius of circle in meters
        num_points: Number of waypoints along the circle
        
    Returns:
        List of [x, y] target points
    """
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    trajectory = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_square_trajectory(center, side_length, num_points=20):
    """
    Generate a square trajectory.
    
    Args:
        center: Center point [x, y] in meters
        side_length: Length of each side in meters
        num_points: Number of waypoints along the square (distributed evenly)
        
    Returns:
        List of [x, y] target points
    """
    half_side = side_length / 2
    points_per_side = num_points // 4
    trajectory = []
    
    # Top side (left to right)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] - half_side + side_length * t
        y = center[1] + half_side
        trajectory.append(np.array([x, y]))
    
    # Right side (top to bottom)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] + half_side
        y = center[1] + half_side - side_length * t
        trajectory.append(np.array([x, y]))
    
    # Bottom side (right to left)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] + half_side - side_length * t
        y = center[1] - half_side
        trajectory.append(np.array([x, y]))
    
    # Left side (bottom to top)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] - half_side
        y = center[1] - half_side + side_length * t
        trajectory.append(np.array([x, y]))
    
    return trajectory


def generate_figure_eight_trajectory(center, width, height, num_points=40):
    """
    Generate a figure-eight (infinity symbol) trajectory.
    
    Args:
        center: Center point [x, y] in meters
        width: Width of the figure-eight in meters
        height: Height of the figure-eight in meters
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    trajectory = []
    for angle in t:
        # Parametric equations for figure-eight
        x = center[0] + (width/2) * np.sin(angle)
        y = center[1] + (height/2) * np.sin(angle) * np.cos(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_spiral_trajectory(center, max_radius, num_loops=3, num_points=60):
    """
    Generate an outward spiral trajectory.
    
    Args:
        center: Center point [x, y] in meters
        max_radius: Maximum radius of spiral in meters
        num_loops: Number of complete loops
        num_points: Total number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    for i in range(num_points):
        t = i / num_points
        angle = t * num_loops * 2 * np.pi
        radius = max_radius * t
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_zigzag_trajectory(start, end, amplitude, num_points=20):
    """
    Generate a zigzag trajectory from start to end point.
    
    Args:
        start: Starting point [x, y] in meters
        end: Ending point [x, y] in meters
        amplitude: Amplitude of zigzag perpendicular to main direction
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    direction = np.array(end) - np.array(start)
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else perpendicular
    
    for i in range(num_points):
        t = i / (num_points - 1)
        # Main direction progress
        base_point = np.array(start) + t * direction
        # Zigzag offset
        offset = amplitude * np.sin(t * np.pi * 4) * perpendicular
        trajectory.append(base_point + offset)
    return trajectory


def generate_star_trajectory(center, outer_radius, inner_radius, num_points=10):
    """
    Generate a star-shaped trajectory.
    
    Args:
        center: Center point [x, y] in meters
        outer_radius: Radius of outer points in meters
        inner_radius: Radius of inner points in meters
        num_points: Number of points (must be even, half for outer, half for inner)
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    for i in range(num_points):
        angle = i / num_points * 2 * np.pi
        # Alternate between outer and inner radius
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_sine_wave_trajectory(start, end, amplitude, frequency=2, num_points=30):
    """
    Generate a sine wave trajectory.
    
    Args:
        start: Starting point [x, y] in meters
        end: Ending point [x, y] in meters
        amplitude: Amplitude of the wave
        frequency: Number of complete waves
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    direction = np.array(end) - np.array(start)
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else perpendicular
    
    for i in range(num_points):
        t = i / (num_points - 1)
        base_point = np.array(start) + t * direction
        offset = amplitude * np.sin(t * 2 * np.pi * frequency) * perpendicular
        trajectory.append(base_point + offset)
    return trajectory


def test_trajectory_tracking(env, model, trajectory, steps_per_target=50, render=True):
    """
    Test the robot's ability to track a trajectory.
    
    Args:
        env: The environment
        model: The trained model
        trajectory: List of target points
        steps_per_target: Number of steps to attempt reaching each target
        render: Whether to render the environment
        
    Returns:
        Dictionary with tracking statistics
    """
    obs, _ = env.reset()
    
    # Set the full trajectory for visualization
    # env.set_trajectory(trajectory)
    
    # Position robot at the first waypoint with orientation toward the second
    if len(trajectory) >= 2:
        start_pos = trajectory[0]
        next_pos = trajectory[1]
        
        # Set robot position to first waypoint
        env.robot.position[0] = start_pos[0]
        env.robot.position[1] = start_pos[1]
        
        # Calculate orientation angle from first to second waypoint
        direction = next_pos - start_pos
        yaw_angle = np.arctan2(direction[1], direction[0])
        env.robot.euler_angle[2] = yaw_angle  # Set yaw
        
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"Orientation toward waypoint 1: {np.degrees(yaw_angle):.1f}°")
    elif len(trajectory) == 1:
        # Only one waypoint, just position at it
        start_pos = trajectory[0]
        env.robot.position[0] = start_pos[0]
        env.robot.position[1] = start_pos[1]
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    
    total_steps = 0
    targets_reached = 0
    distances_to_targets = []
    
    trajectory.append(trajectory[0])  # Loop back to start for continuous tracking
    trajectory = trajectory[1:]
    env.set_trajectory(trajectory) 
    for target_idx, target in enumerate(trajectory):
        # Update current waypoint index for visualization
        env.current_waypoint_index = target_idx
        # Set the new target in the environment
        env.target_point = target
        print(f"\nTarget {target_idx+1}/{len(trajectory)}: ({target[0]:.2f}, {target[1]:.2f})")
        
        min_distance = float('inf')
        
        for step in range(steps_per_target):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if render:
                env.wait_for_animation()
            
            # Calculate distance to current target
            distance = np.linalg.norm(env.robot.position[0:-1] - target)
            min_distance = min(min_distance, distance)
            
            total_steps += 1
            
            if distance < 0.05:  # Threshold for "reaching" the target
                targets_reached += 1
                print(f"  ✓ Reached in {step+1} steps (distance: {distance:.3f}m)")
                break
            
            if truncated or terminated:
                obs, _ = env.reset()
                # Re-set the trajectory after reset
                env.set_trajectory(trajectory)
                env.current_waypoint_index = target_idx
                env.target_point = target
                print(f"  Reset environment (truncated={truncated}, terminated={terminated})")
        
        distances_to_targets.append(min_distance)
        if min_distance >= 0.05:
            print(f"  ✗ Closest approach: {min_distance:.3f}m")
    
    stats = {
        'total_targets': len(trajectory),
        'targets_reached': targets_reached,
        'success_rate': targets_reached / len(trajectory),
        'avg_min_distance': np.mean(distances_to_targets),
        'total_steps': total_steps
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    env = SalpRobotEnv(render_mode="human", robot=robot)
    
    # Load the trained model
    model = SAC.load("./logs/salp_robot_model_400000_steps", env=env)   
    
    # Choose a trajectory type
    center = np.array([0.0, 0.0])
    
    # Test different trajectories
    trajectories = {
        'circle': generate_circle_trajectory(center, radius=1.0, num_points=16),
        'square': generate_square_trajectory(center, side_length=2.0, num_points=20),
        'figure_eight': generate_figure_eight_trajectory(center, width=2.0, height=1.0, num_points=30),
        'spiral': generate_spiral_trajectory(center, max_radius=1.0, num_loops=2, num_points=40),
        'star': generate_star_trajectory(center, outer_radius=1.0, inner_radius=0.5, num_points=10),
        'sine_wave': generate_sine_wave_trajectory(
            start=np.array([-1.0, 0.0]), 
            end=np.array([1.0, 0.0]), 
            amplitude=0.5, 
            frequency=3,
            num_points=25
        )
    }
    
    # Select which trajectory to test (change this to test different shapes)
    trajectory_name = 'figure_eight'  # Options: circle, square, figure_eight, spiral, star, sine_wave
    trajectory = trajectories[trajectory_name]
    
    print(f"\n{'='*60}")
    print(f"Testing {trajectory_name.upper()} trajectory")
    print(f"{'='*60}")
    
    env.start_recording()
    
    # Test the trajectory
    stats = test_trajectory_tracking(env, model, trajectory, steps_per_target=100, render=True)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TRAJECTORY TRACKING RESULTS - {trajectory_name.upper()}")
    print(f"{'='*60}")
    print(f"Total targets: {stats['total_targets']}")
    print(f"Targets reached: {stats['targets_reached']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Average minimum distance: {stats['avg_min_distance']:.3f}m")
    print(f"Total steps: {stats['total_steps']}")

    gif_path = env.stop_recording(f"trajectory_{trajectory_name}_test.gif")
    env.close()
