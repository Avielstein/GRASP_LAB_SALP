"""
SALP Robot Simulation
Bio-inspired soft underwater robot with steerable rear nozzle.
Based on research from University of Pennsylvania Sung Robotics Lab.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class Robot():
    phase = ["contract", "release", "coast"]

    def __init__(self, dry_mass: float, init_length: float, init_width: float, max_contraction: float, nozzle_area: float):
        self.dry_mass = dry_mass # kg
        self.init_length = init_length # meters
        self.init_width = init_width # meters
        self.max_contraction = max_contraction  # max contraction length
        self.state = "rest"
        self.contraciton = 0.0  # contraction level
        self.angle = 0.0  # current angle of nozzle
        self.cycle = 0
        self.dt = 0.01
        self.time = 0.0
        self.cycle_time = 0.0
        self.positions = np.zeros(3)  # x, y, z positions
        self.velocities = np.zeros(3)  # x, y, z velocities
        self.previous_water_volume = 0.0
        self.nozzle_area = 0.0001  # m^2, cross-sectional area of the nozzle

    def set_control(self, contraction: float, angle: float):

        self.contraciton = contraction
        self.angle = angle
        self.cycle += 1 

    def get_state(self) -> str:

        contract_time = self._contract_model()
        release_time = self._release_model()

        if self.cycle_time < contract_time:
            self.state = self.phase[0]  # contract
        elif self.cycle_time < contract_time + release_time:
            self.state = self.phase[1]  # release
        else:
            self.state = self.phase[2]  # coast
        return self.state

    def step(self, time: float):

        self.get_state()
        if self.state == self.phase[0]:  # contract
            self.contract(inhale=True) 
        elif self.state == self.phase[1]:  # release
            self.release() 
        else:
            self.coasting()

    def contract(self):
        # computes
        # a = F/m
        # v = v + a*dt
        # x = x + v*dt

        self._get_jet_force()
        self.get_mass()
        a = self.jet_force / self.mass  # acceleration
        self.velocities += a * self.dt  # update velocities
        self.positions += self.velocities * self.dt  # update positions

    def release(self):
        pass

    def coasting(self):
        pass
    
    def _get_jet_force(self) -> float:

        water_mass = self._get_water_volume()
        mass_rate = (water_mass - self.previous_water_volume) / self.dt
        jet_velocity = self._get_jet_velocity()
        jet_force = mass_rate * jet_velocity

        return jet_force
    
    def _get_jet_velocity(self) -> float:
        
        # velocity is with respect to the robot frame
        water_volume = self._get_water_volume()
        volume_rate = (water_volume - self.previous_water_volume) / self.dt
        jet_velocity = volume_rate / self.nozzle_area

        return jet_velocity
    
    def _get_drag_force(self) -> float:



    def _get_current_length(self, time) -> float:

        if self.state == self.phase[0]:  # inhale
            length = self._contract_model(time)
        elif self.state == self.phase[1]:  # exhale
            length = self._release_model(time)
        else:
            length = self.init_length

        return length
    
    def _get_current_width(self) -> float:

        if self.state == self.phase[0]:  # inhale
            width = self._contract_model(time)
        elif self.state == self.phase[1]:  # exhale
            width = self._release_model(time)
        else:
            width = self.init_width

        return width

    def _get_water_volume(self) -> float:
        
        length = self._get_current_length()
        width = self._get_current_width()
        volume = 4/3*np.pi*(length/2)*(width/2)**2

        return volume

    def _get_water_mass(self) -> float:

        density = 997  # kg/m^3
        mass = density * self._get_water_volume()

        return mass

    def _contract_model(self) -> float:
        # Simple model for contraction over time
        rate = 0.06/3  # m/s
        time = self.contraciton / rate
        return time

    def _release_model(self) -> float:
        # Simple model for release over time
        rate = 0.06/1.5  # m/s
        time = self.contraciton / rate
        return time 


    # def _test_compression_speed(self):
    #     # function is designed to model a constant force presses on a spring 
    #     # with a mass on it
    #     F = 1  # N
    #     k = 1  # N/m
    #     m = 70   # kg
    #     T = 5  # s
    #     n = 500  # steps
    #     x = np.zeros(n)  # m # displacement
    #     v = np.zeros(n)  # m/s # velocity
    #     a = np.zeros(n)  # m/s^2 # acceleration
    #     dt = T / n  # s # time step
    #     for i in range(n-1):
    #         a[i] = (F - k*x[i]) / m
    #         v[i+1] = v[i] + a[i]*dt
    #         x[i+1] = x[i] + v[i]*dt

    #     plt.plot(np.arange(0, T, dt), x*1000, label='Displacement')
    #     # plt.plot(np.arange(0, T, dt), v, color='orange', label='Velocity')
    #     # plt.plot(np.arange(0, T, dt), a, color='green', label='Acceleration')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Displacement (mm)')
    #     plt.title('Compression Speed Test')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

class SalpRobotEnv(gym.Env):
    """
    SALP-inspired robot environment with steerable nozzle.
    
    Features:
    - Slow, realistic breathing cycles (2-3 seconds per phase)
    - Hold-to-inhale control scheme
    - Steerable rear nozzle (not body rotation)
    - Realistic underwater physics and momentum
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 800, height: int = 600):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.tank_margin = 50
        
        # SALP robot parameters
        self.base_radius = 30  # Base body radius (circle at rest)
        self.max_thrust_force = 100  # Thrust per expansion
        self.drag_coefficient = 0.98  # Underwater drag
        
        # Constant surface area parameters
        # Circle surface area = 2πr, Ellipse surface area ≈ π(a + b)
        # For constant surface area: 2πr = π(a + b), so a + b = 2r
        self.base_surface_area = 2 * math.pi * self.base_radius
        
        # Nozzle parameters
        self.max_nozzle_angle = math.pi / 3  # ±60 degrees nozzle steering
        self.nozzle_response_rate = 0.05  # How fast nozzle moves
        
        # Realistic breathing cycle parameters (much slower)
        self.inhale_duration = 120  # 2 seconds at 60fps
        self.exhale_duration = 150  # 2.5 seconds at 60fps
        self.rest_duration = 60    # 1 second rest between cycles
        
        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Robot state
        self.robot_pos = np.array([width/2, height/2], dtype=float)
        self.robot_velocity = np.array([0.0, 0.0], dtype=float)
        self.robot_angle = 0.0  # Body orientation (changes slowly due to physics)
        self.robot_angular_velocity = 0.0
        
        # Nozzle state
        self.nozzle_angle = 0.0  # Relative to body orientation (-max_nozzle_angle to +max_nozzle_angle)
        self.target_nozzle_angle = 0.0
        
        # Breathing state
        self.breathing_phase = "rest"  # "rest", "inhaling", "exhaling"
        self.breathing_timer = 0
        self.body_radius = self.base_radius  # Current body radius
        self.ellipse_a = self.base_radius    # Semi-major axis for ellipse
        self.ellipse_b = self.base_radius    # Semi-minor axis for ellipse
        self.is_inhaling = False  # True when space is held down
        self.water_volume = 0.0  # Amount of water inhaled (0-1)
        
        # Action space: [inhale_control (0/1), nozzle_direction (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [pos_x, pos_y, vel_x, vel_y, body_angle, angular_vel, body_size, breathing_phase, water_volume, nozzle_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -math.pi, -0.1, 0.5, 0, 0, -1]),
            high=np.array([width, height, 10, 10, math.pi, 0.1, 2.0, 2, 1, 1]),
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
        
        # Reset nozzle
        self.nozzle_angle = 0.0
        self.target_nozzle_angle = 0.0
        
        # Reset breathing state
        self.breathing_phase = "rest"
        self.breathing_timer = 0
        self.body_radius = self.base_radius  # Current body radius
        self.ellipse_a = self.base_radius    # Semi-major axis for ellipse
        self.ellipse_b = self.base_radius    # Semi-minor axis for ellipse
        self.is_inhaling = False
        self.water_volume = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Check if we're in forced breathing mode (passed from environment)
        forced_breathing = getattr(self, 'forced_breathing', False)
        
        if forced_breathing:
            # FORCED BREATHING MODE: Only nozzle control
            nozzle_direction = float(action[0])  # -1 to 1 (single action)
            # Force automatic breathing cycle
            self.is_inhaling = self._get_forced_breathing_state()
        else:
            # NORMAL MODE: Full control
            inhale_control = float(action[0])  # 0 to 1
            nozzle_direction = float(action[1])  # -1 to 1
            self.is_inhaling = inhale_control > 0.5
        
        # Update control inputs
        self.target_nozzle_angle = nozzle_direction * self.max_nozzle_angle
        
        # Update nozzle position (smooth movement)
        self._update_nozzle()
        
        # Update breathing cycle
        self._update_breathing_cycle()
        
        # Update physics
        self._update_physics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = False
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _get_forced_breathing_state(self) -> bool:
        """Determine breathing state for forced breathing mode."""
        # Create automatic breathing cycle based on timer
        cycle_length = self.inhale_duration + self.exhale_duration + self.rest_duration
        cycle_position = self.breathing_timer % cycle_length
        
        if cycle_position < self.inhale_duration:
            return True  # Inhaling
        else:
            return False  # Exhaling or resting
    
    def _update_nozzle(self):
        """Update nozzle angle smoothly."""
        angle_diff = self.target_nozzle_angle - self.nozzle_angle
        if abs(angle_diff) > self.nozzle_response_rate:
            if angle_diff > 0:
                self.nozzle_angle += self.nozzle_response_rate
            else:
                self.nozzle_angle -= self.nozzle_response_rate
        else:
            self.nozzle_angle = self.target_nozzle_angle
        
        # Clamp nozzle angle
        self.nozzle_angle = max(-self.max_nozzle_angle, 
                              min(self.max_nozzle_angle, self.nozzle_angle))
    
    def _update_breathing_cycle(self):
        """Update the breathing cycle with correct sequence: ellipsoid → sphere → ellipsoid."""
        if self.breathing_phase == "rest":
            # At rest: ellipsoid shape (natural resting state)
            self.body_radius = self.base_radius
            # Start with moderate ellipsoid shape
            self.ellipse_a = self.base_radius * 1.3  # Slightly elongated
            self.ellipse_b = self.base_radius * 0.8  # Slightly compressed
            
            if self.is_inhaling:
                # Start inhaling
                self.breathing_phase = "inhaling"
                self.breathing_timer = 0
            
        elif self.breathing_phase == "inhaling":
            if self.is_inhaling and self.breathing_timer < self.inhale_duration:
                # Continue inhaling - body becomes more spherical (filling with water)
                self.breathing_timer += 1
                progress = self.breathing_timer / self.inhale_duration
                
                # Transition from ellipsoid to sphere as water fills
                # Start: ellipse_a = 1.3*r, ellipse_b = 0.8*r
                # End: ellipse_a = 1.1*r, ellipse_b = 1.1*r (slightly larger sphere)
                start_a = self.base_radius * 1.3
                start_b = self.base_radius * 0.8
                end_a = self.base_radius * 1.1  # Slightly expanded sphere
                end_b = self.base_radius * 1.1
                
                self.ellipse_a = start_a + (end_a - start_a) * progress
                self.ellipse_b = start_b + (end_b - start_b) * progress
                
                self.water_volume = progress
                
            else:
                # Stop inhaling (either released early or reached max inhale)
                if self.water_volume > 0.05:  # Only exhale if we inhaled some water
                    self.breathing_phase = "exhaling"
                    self.breathing_timer = 0
                    # Scale exhale duration based on how much water was inhaled
                    self.current_exhale_duration = int(self.exhale_duration * max(self.water_volume, 0.3))
                else:
                    # Return to rest if barely inhaled (early release with minimal water)
                    self.breathing_phase = "rest"
                    self.breathing_timer = 0
                    self.water_volume = 0.0
        
        elif self.breathing_phase == "exhaling":
            self.breathing_timer += 1
            # Use scaled exhale duration based on water volume
            exhale_duration = getattr(self, 'current_exhale_duration', self.exhale_duration)
            progress = self.breathing_timer / exhale_duration
            
            if progress <= 1.0:
                # Exhale - body transitions from sphere back to ellipsoid (expelling water)
                # Start: sphere (ellipse_a = 1.1*r, ellipse_b = 1.1*r)
                # End: ellipsoid (ellipse_a = 1.3*r, ellipse_b = 0.8*r)
                start_a = self.base_radius * 1.1
                start_b = self.base_radius * 1.1
                end_a = self.base_radius * 1.3
                end_b = self.base_radius * 0.8
                
                self.ellipse_a = start_a + (end_a - start_a) * progress
                self.ellipse_b = start_b + (end_b - start_b) * progress
                
                # Apply thrust during early expansion, scaled by water volume
                if 0.1 <= progress <= 0.5:
                    self._apply_jet_thrust()
                
                # Reduce water volume
                self.water_volume = max(0, self.water_volume * (1.0 - progress))
                
            else:
                # Exhale complete, return to rest
                self.breathing_phase = "rest"
                self.breathing_timer = 0
                self.water_volume = 0.0
    
    def _apply_jet_thrust(self):
        """Apply jet thrust through steerable nozzle with improved moment physics."""
        # Calculate thrust based on water volume and expansion rate
        thrust_magnitude = self.max_thrust_force * self.water_volume * 0.4
        
        # Thrust direction: nozzle points backward, thrust pushes forward
        thrust_angle = self.robot_angle - self.nozzle_angle
        
        thrust_x = math.cos(thrust_angle) * thrust_magnitude
        thrust_y = math.sin(thrust_angle) * thrust_magnitude
        
        # Apply thrust to velocity
        self.robot_velocity[0] += thrust_x * 0.012
        self.robot_velocity[1] += thrust_y * 0.012
        
        # Torque physics: angled nozzle creates rotation
        # When nozzle points right (+), robot should turn left (-)
        primary_torque = -self.nozzle_angle * thrust_magnitude * 0.0002
        
        # Moment arm effect - thrust applied at rear creates rotation
        moment_arm = max(self.ellipse_a, self.ellipse_b) * 0.7
        thrust_perpendicular = thrust_magnitude * math.sin(-self.nozzle_angle)
        moment_torque = thrust_perpendicular * moment_arm * 0.00005
        
        # Body shape effect during morphing
        shape_torque = -self.nozzle_angle * thrust_magnitude * self.water_volume * 0.00003
        
        # Combine torque effects
        total_torque = primary_torque + moment_torque + shape_torque
        self.robot_angular_velocity += total_torque
        
        # Thrust vectoring effect - angled nozzle creates side force
        side_thrust_angle = thrust_angle + math.pi/2
        side_thrust_magnitude = thrust_magnitude * abs(self.nozzle_angle) * 0.3
        
        side_thrust_x = math.cos(side_thrust_angle) * side_thrust_magnitude
        side_thrust_y = math.sin(side_thrust_angle) * side_thrust_magnitude
        
        self.robot_velocity[0] += side_thrust_x * 0.008
        self.robot_velocity[1] += side_thrust_y * 0.008
        
        # Add slight random variation for realism
        noise_angle = thrust_angle + (np.random.random() - 0.5) * 0.05
        noise_force = thrust_magnitude * 0.04
        self.robot_velocity[0] += math.cos(noise_angle) * noise_force * 0.002
        self.robot_velocity[1] += math.sin(noise_angle) * noise_force * 0.002
    
    def _update_physics(self):
        """Update robot physics with realistic underwater dynamics."""
        # Apply drag
        self.robot_velocity *= self.drag_coefficient
        self.robot_angular_velocity *= 0.95
        
        # Update position
        self.robot_pos += self.robot_velocity
        
        # Update body angle (changes slowly due to momentum and thrust)
        self.robot_angle += self.robot_angular_velocity
        
        # Normalize angle
        while self.robot_angle > math.pi:
            self.robot_angle -= 2 * math.pi
        while self.robot_angle < -math.pi:
            self.robot_angle += 2 * math.pi
        
        # Keep robot within bounds (bounce off walls)
        margin = self.tank_margin + max(self.ellipse_a, self.ellipse_b)
        if self.robot_pos[0] < margin:
            self.robot_pos[0] = margin
            self.robot_velocity[0] = abs(self.robot_velocity[0]) * 0.4
            self.robot_angular_velocity *= 0.7
        elif self.robot_pos[0] > self.width - margin:
            self.robot_pos[0] = self.width - margin
            self.robot_velocity[0] = -abs(self.robot_velocity[0]) * 0.4
            self.robot_angular_velocity *= 0.7
        
        if self.robot_pos[1] < margin:
            self.robot_pos[1] = margin
            self.robot_velocity[1] = abs(self.robot_velocity[1]) * 0.4
            self.robot_angular_velocity *= 0.7
        elif self.robot_pos[1] > self.height - margin:
            self.robot_pos[1] = self.height - margin
            self.robot_velocity[1] = -abs(self.robot_velocity[1]) * 0.4
            self.robot_angular_velocity *= 0.7
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on realistic movement and efficiency."""
        # Reward for smooth movement
        speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        movement_reward = min(speed * 0.08, 0.6)
        
        # Reward for efficient breathing (not too frequent)
        breathing_efficiency = 0.15 if self.breathing_phase == "rest" else 0.08
        
        # Small penalty for excessive nozzle movement (energy cost)
        nozzle_penalty = abs(self.nozzle_angle) * 0.02
        
        # Small reward for staying in bounds
        bounds_reward = 0.05
        
        return movement_reward + breathing_efficiency + bounds_reward - nozzle_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Map breathing phase to number
        phase_map = {"rest": 0, "inhaling": 1, "exhaling": 2}
        phase_num = phase_map.get(self.breathing_phase, 0)
        
        return np.array([
            self.robot_pos[0] / self.width,  # Normalized position
            self.robot_pos[1] / self.height,
            self.robot_velocity[0] / 5.0,  # Normalized velocity
            self.robot_velocity[1] / 5.0,
            self.robot_angle / math.pi,  # Normalized body angle
            self.robot_angular_velocity / 0.1,  # Normalized angular velocity
            max(self.ellipse_a, self.ellipse_b) / self.base_radius,  # Normalized body size
            phase_num / 2.0,  # Normalized breathing phase
            self.water_volume,  # Water volume (0-1)
            self.nozzle_angle / self.max_nozzle_angle  # Normalized nozzle angle
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'robot_position': tuple(self.robot_pos),
            'robot_velocity': tuple(self.robot_velocity),
            'robot_angle': self.robot_angle,
            'nozzle_angle': self.nozzle_angle,
            'ellipse_a': self.ellipse_a,
            'ellipse_b': self.ellipse_b,
            'body_radius': self.body_radius,
            'breathing_phase': self.breathing_phase,
            'water_volume': self.water_volume,
            'breathing_timer': self.breathing_timer
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            
            if self.width <= 0 or self.height <= 0:
                self.width = 800
                self.height = 600
            
            if self.render_mode == "human":
                try:
                    self.screen = pygame.display.set_mode((int(self.width), int(self.height)))
                    pygame.display.set_caption("SALP Robot")
                except pygame.error as e:
                    print(f"Pygame display error: {e}")
                    self.width, self.height = 640, 480
                    self.screen = pygame.display.set_mode((self.width, self.height))
            else:
                self.screen = pygame.Surface((int(self.width), int(self.height)))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen with deep water color
        self.screen.fill((10, 25, 50))
        
        # Draw tank boundaries
        pygame.draw.rect(self.screen, (30, 60, 100), 
                        (self.tank_margin, self.tank_margin, 
                         self.width - 2*self.tank_margin, self.height - 2*self.tank_margin), 3)
        
        # Draw robot
        robot_x, robot_y = int(self.robot_pos[0]), int(self.robot_pos[1])
        
        # Body color based on breathing phase
        phase_colors = {
            "rest": (100, 140, 180),
            "inhaling": (70, 100, 150),
            "exhaling": (150, 100, 70)
        }
        body_color = phase_colors.get(self.breathing_phase, (100, 140, 180))
        
        # Draw morphing body
        ellipse_width = int(self.ellipse_a * 2)
        ellipse_height = int(self.ellipse_b * 2)
        
        ellipse_surf = pygame.Surface((ellipse_width, ellipse_height), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surf, body_color, (0, 0, ellipse_width, ellipse_height))
        
        rotated_surf = pygame.transform.rotate(ellipse_surf, -math.degrees(self.robot_angle))
        rect = rotated_surf.get_rect(center=(robot_x, robot_y))
        self.screen.blit(rotated_surf, rect)
        
        # Draw body outline
        outline_radius = int(max(self.ellipse_a, self.ellipse_b))
        pygame.draw.circle(self.screen, (60, 80, 120), (robot_x, robot_y), outline_radius, 2)
        
        # Draw front indicator
        front_distance = max(self.ellipse_a, self.ellipse_b) * 0.8
        front_x = robot_x + math.cos(self.robot_angle) * front_distance
        front_y = robot_y + math.sin(self.robot_angle) * front_distance
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 4)
        
        # Draw steerable nozzle
        back_distance = max(self.ellipse_a, self.ellipse_b) * 0.9
        back_x = robot_x + math.cos(self.robot_angle + math.pi) * back_distance
        back_y = robot_y + math.sin(self.robot_angle + math.pi) * back_distance
        
        nozzle_world_angle = self.robot_angle + math.pi + self.nozzle_angle
        nozzle_length = 15
        nozzle_end_x = back_x + math.cos(nozzle_world_angle) * nozzle_length
        nozzle_end_y = back_y + math.sin(nozzle_world_angle) * nozzle_length
        
        pygame.draw.line(self.screen, (200, 200, 100), 
                        (int(back_x), int(back_y)), (int(nozzle_end_x), int(nozzle_end_y)), 4)
        
        # Draw water jet during exhale
        if self.breathing_phase == "exhaling":
            num_particles = 8
            for i in range(num_particles):
                base_distance = nozzle_length + 5 + i * 4
                curve_factor = abs(self.nozzle_angle) * 0.5
                curve_offset = curve_factor * (i * 0.3)
                
                perpendicular_angle = nozzle_world_angle + math.pi/2
                if self.nozzle_angle > 0:
                    curve_offset = -curve_offset
                
                straight_x = back_x + math.cos(nozzle_world_angle) * base_distance
                straight_y = back_y + math.sin(nozzle_world_angle) * base_distance
                
                curved_x = straight_x + math.cos(perpendicular_angle) * curve_offset
                curved_y = straight_y + math.sin(perpendicular_angle) * curve_offset
                
                spread_variation = (i - num_particles/2) * 0.08
                spread_x = curved_x + math.cos(perpendicular_angle) * spread_variation * 3
                spread_y = curved_y + math.sin(perpendicular_angle) * spread_variation * 3
                
                particle_size = max(1, 5 - i)
                blue_intensity = max(100, 200 - i * 15)
                particle_color = (80, 120, blue_intensity)
                
                pygame.draw.circle(self.screen, particle_color, 
                                 (int(spread_x), int(spread_y)), particle_size)
        
        # Draw velocity vector
        speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        if speed > 0.5:
            vel_scale = 8
            vel_end_x = robot_x + self.robot_velocity[0] * vel_scale
            vel_end_y = robot_y + self.robot_velocity[1] * vel_scale
            pygame.draw.line(self.screen, (0, 255, 150), 
                           (robot_x, robot_y), (int(vel_end_x), int(vel_end_y)), 2)
        
        # UI information
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            info_lines = [
                "SALP Robot - Steerable Nozzle",
                f"Phase: {self.breathing_phase.title()}",
                f"Body Size: {max(self.ellipse_a, self.ellipse_b):.1f}",
                f"Water: {self.water_volume:.2f}",
                f"Speed: {speed:.1f}",
                f"Nozzle: {math.degrees(self.nozzle_angle):.0f}°",
            ]
            
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
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


if __name__ == "__main__":
    
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.1)
    robot._test_compression_speed()