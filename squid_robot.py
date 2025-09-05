"""
Simple Squid Robot Simulation
A clean implementation focusing on basic squid motion mechanics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Optional, Dict, Any


class SquidRobotEnv(gym.Env):
    """
    A simple squid robot environment focused on basic motion mechanics.
    
    Features:
    - Top-down 2D view
    - Jet propulsion with steerable nozzle
    - Water management system
    - Clean tank with no obstacles
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 800, height: int = 600):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.tank_margin = 50
        
        # Squid robot parameters
        self.robot_size = 25  # Robot radius
        self.max_nozzle_angle = math.pi / 4  # ±45 degrees
        self.max_thrust_force = 150  # Maximum jet thrust
        self.water_capacity = 100  # Water tank capacity
        self.refill_rate = 10  # Water refill per step when not thrusting
        self.thrust_consumption = 8  # Water consumed per thrust pulse
        self.drag_coefficient = 0.98  # Velocity damping (closer to 1 = less drag)
        
        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Robot state
        self.robot_pos = np.array([width/2, height/2], dtype=float)
        self.robot_velocity = np.array([0.0, 0.0], dtype=float)
        self.robot_angle = 0.0  # Robot orientation
        self.robot_angular_velocity = 0.0
        self.water_level = 0  # Current water in tank
        self.nozzle_angle = 0.0  # Current nozzle angle
        self.thrust_cooldown = 0  # Cooldown between thrusts
        
        # Action space: [thrust_power (0-1), nozzle_angle (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, water_level, nozzle_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -50, -50, -math.pi, -3, 0, -1]),
            high=np.array([width, height, 50, 50, math.pi, 3, 100, 1]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset robot to center
        self.robot_pos = np.array([self.width/2, self.height/2], dtype=float)
        self.robot_velocity = np.array([0.0, 0.0], dtype=float)
        self.robot_angle = 0.0
        self.robot_angular_velocity = 0.0
        self.water_level = self.water_capacity  # Start with full tank
        self.nozzle_angle = 0.0
        self.thrust_cooldown = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        thrust_power = float(action[0])  # 0 to 1
        nozzle_target = float(action[1])  # -1 to 1
        
        # Update nozzle angle smoothly
        target_angle = nozzle_target * self.max_nozzle_angle
        self.nozzle_angle += (target_angle - self.nozzle_angle) * 0.2
        
        # Handle thrust system
        if thrust_power > 0.1 and self.water_level >= self.thrust_consumption and self.thrust_cooldown <= 0:
            self._apply_thrust(thrust_power)
            self.water_level -= self.thrust_consumption
            self.thrust_cooldown = 6  # Cooldown period
        else:
            # Refill water tank when not thrusting
            if self.water_level < self.water_capacity:
                self.water_level = min(self.water_capacity, self.water_level + self.refill_rate)
        
        # Decrease cooldown
        if self.thrust_cooldown > 0:
            self.thrust_cooldown -= 1
        
        # Update physics
        self._update_physics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination (none for now - continuous environment)
        done = False
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _apply_thrust(self, thrust_power: float):
        """Apply jet propulsion thrust with pulsing jellyfish-like motion."""
        # Calculate thrust direction (opposite to nozzle direction)
        thrust_angle = self.robot_angle + math.pi + self.nozzle_angle
        
        # Create pulsing effect - like a balloon deflating/inflating
        pulse_phase = (60 - self.thrust_cooldown) / 6.0  # 0 to 10 over cooldown period
        pulse_intensity = 0.5 + 0.5 * math.sin(pulse_phase * math.pi)  # Wave from 0 to 1
        
        # Calculate thrust force with pulsing
        base_thrust = thrust_power * self.max_thrust_force
        pulsed_thrust = base_thrust * pulse_intensity
        thrust_x = math.cos(thrust_angle) * pulsed_thrust
        thrust_y = math.sin(thrust_angle) * pulsed_thrust
        
        # Apply thrust to velocity with wave-like motion
        self.robot_velocity[0] += thrust_x * 0.012
        self.robot_velocity[1] += thrust_y * 0.012
        
        # Jellyfish-like drift: nozzle at back causes body to drift/wobble
        drift_angle = self.robot_angle + math.pi/2  # Perpendicular to robot direction
        drift_force = pulsed_thrust * 0.3 * math.sin(pulse_phase * 2)  # Side drift
        self.robot_velocity[0] += math.cos(drift_angle) * drift_force * 0.003
        self.robot_velocity[1] += math.sin(drift_angle) * drift_force * 0.003
        
        # Enhanced steering torque with pulsing effect
        torque = self.nozzle_angle * pulsed_thrust * 0.0001
        self.robot_angular_velocity += torque
        
        # Add slight rotational wobble like jellyfish
        wobble = 0.02 * math.sin(pulse_phase * 3) * thrust_power
        self.robot_angular_velocity += wobble
    
    def _update_physics(self):
        """Update robot physics."""
        # Apply drag
        self.robot_velocity *= self.drag_coefficient
        self.robot_angular_velocity *= 0.95
        
        # Update position
        self.robot_pos += self.robot_velocity
        
        # Update angle
        self.robot_angle += self.robot_angular_velocity
        
        # Normalize angle
        while self.robot_angle > math.pi:
            self.robot_angle -= 2 * math.pi
        while self.robot_angle < -math.pi:
            self.robot_angle += 2 * math.pi
        
        # Keep robot within bounds (bounce off walls)
        if self.robot_pos[0] < self.tank_margin:
            self.robot_pos[0] = self.tank_margin
            self.robot_velocity[0] = abs(self.robot_velocity[0]) * 0.3
        elif self.robot_pos[0] > self.width - self.tank_margin:
            self.robot_pos[0] = self.width - self.tank_margin
            self.robot_velocity[0] = -abs(self.robot_velocity[0]) * 0.3
        
        if self.robot_pos[1] < self.tank_margin:
            self.robot_pos[1] = self.tank_margin
            self.robot_velocity[1] = abs(self.robot_velocity[1]) * 0.3
        elif self.robot_pos[1] > self.height - self.tank_margin:
            self.robot_pos[1] = self.height - self.tank_margin
            self.robot_velocity[1] = -abs(self.robot_velocity[1]) * 0.3
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on movement and efficiency."""
        # Reward for movement
        speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        movement_reward = min(speed * 0.1, 1.0)
        
        # Reward for water efficiency
        water_efficiency = self.water_level / self.water_capacity
        efficiency_reward = water_efficiency * 0.3
        
        # Small reward for staying in bounds
        bounds_reward = 0.1
        
        return movement_reward + efficiency_reward + bounds_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.robot_pos[0] / self.width,  # Normalized position
            self.robot_pos[1] / self.height,
            self.robot_velocity[0] / 10.0,  # Normalized velocity
            self.robot_velocity[1] / 10.0,
            self.robot_angle / math.pi,  # Normalized angle
            self.robot_angular_velocity / 2.0,  # Normalized angular velocity
            self.water_level / self.water_capacity,  # Water level (0-1)
            self.nozzle_angle / self.max_nozzle_angle  # Normalized nozzle angle
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'robot_position': tuple(self.robot_pos),
            'robot_velocity': tuple(self.robot_velocity),
            'robot_angle': self.robot_angle,
            'water_level': self.water_level,
            'nozzle_angle': self.nozzle_angle,
            'thrust_cooldown': self.thrust_cooldown
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Simple Squid Robot")
            else:
                self.screen = pygame.Surface((self.width, self.height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((30, 60, 100))  # Dark blue water
        
        # Draw tank boundaries
        pygame.draw.rect(self.screen, (80, 120, 160), 
                        (self.tank_margin, self.tank_margin, 
                         self.width - 2*self.tank_margin, self.height - 2*self.tank_margin), 2)
        
        # Draw robot
        robot_x, robot_y = int(self.robot_pos[0]), int(self.robot_pos[1])
        
        # Robot body
        pygame.draw.circle(self.screen, (200, 120, 60), (robot_x, robot_y), self.robot_size)
        pygame.draw.circle(self.screen, (160, 90, 40), (robot_x, robot_y), self.robot_size, 2)
        
        # Robot front indicator
        front_x = robot_x + math.cos(self.robot_angle) * self.robot_size * 0.7
        front_y = robot_y + math.sin(self.robot_angle) * self.robot_size * 0.7
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 4)
        
        # Nozzle direction
        nozzle_angle_world = self.robot_angle + math.pi + self.nozzle_angle
        nozzle_length = self.robot_size * 1.3
        nozzle_end_x = robot_x + math.cos(nozzle_angle_world) * nozzle_length
        nozzle_end_y = robot_y + math.sin(nozzle_angle_world) * nozzle_length
        
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (robot_x, robot_y), (int(nozzle_end_x), int(nozzle_end_y)), 3)
        
        # Thrust particles
        if self.thrust_cooldown > 0:
            for i in range(3):
                particle_angle = nozzle_angle_world + (i - 1) * 0.2
                particle_distance = nozzle_length + 8 + i * 6
                particle_x = robot_x + math.cos(particle_angle) * particle_distance
                particle_y = robot_y + math.sin(particle_angle) * particle_distance
                
                pygame.draw.circle(self.screen, (120, 180, 255), 
                                 (int(particle_x), int(particle_y)), 3)
        
        # Velocity vector
        if math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2) > 1.0:
            vel_scale = 8
            vel_end_x = robot_x + self.robot_velocity[0] * vel_scale
            vel_end_y = robot_y + self.robot_velocity[1] * vel_scale
            pygame.draw.line(self.screen, (0, 255, 0), 
                           (robot_x, robot_y), (int(vel_end_x), int(vel_end_y)), 2)
        
        # UI information
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            
            info_lines = [
                "Simple Squid Robot",
                f"Water: {self.water_level:.0f}/{self.water_capacity}",
                f"Speed: {math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2):.1f}",
                f"Nozzle: {math.degrees(self.nozzle_angle):.0f}°",
                "W/S: Thrust, A/D: Steer"
            ]
            
            for i, line in enumerate(info_lines):
                color = (255, 255, 255) if i < 4 else (180, 180, 180)
                text = font.render(line, True, color)
                self.screen.blit(text, (10, 10 + i * 25))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


def main():
    """Demo the squid robot environment."""
    print("Simple Squid Robot Demo")
    print("Controls:")
    print("- W/S: Thrust power")
    print("- A/D: Nozzle steering")
    print("- ESC: Quit")
    
    pygame.init()
    env = SquidRobotEnv(render_mode="human")
    observation, info = env.reset()
    
    running = True
    thrust_power = 0.0
    nozzle_angle = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Thrust control
        if keys[pygame.K_w]:
            thrust_power = min(1.0, thrust_power + 0.03)
        elif keys[pygame.K_s]:
            thrust_power = max(0.0, thrust_power - 0.03)
        else:
            thrust_power *= 0.95  # Gradual decay
        
        # Steering control
        if keys[pygame.K_a]:
            nozzle_angle = max(-1.0, nozzle_angle - 0.03)
        elif keys[pygame.K_d]:
            nozzle_angle = min(1.0, nozzle_angle + 0.03)
        else:
            nozzle_angle *= 0.9  # Return to center
        
        # Apply action
        action = np.array([thrust_power, nozzle_angle])
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if done or truncated:
            observation, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    main()
