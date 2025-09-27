"""
Snake-like environment for SALP robot learning.
The SALP must navigate around collecting food items while avoiding collisions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Optional, Dict, Any, List
import random

import sys
import os

# Add project root to path to access scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.salp_robot import SalpRobotEnv


class SalpSnakeEnv(SalpRobotEnv):
    """
    Snake-like environment where SALP collects food items.
    
    Features:
    - Multiple food items scattered around the environment
    - Food respawns when collected
    - Rewards for collecting food
    - Penalties for collisions with walls
    - Time-based penalties to encourage efficiency
    """
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 800, height: int = 600,
                 num_food_items: int = 5, food_reward: float = 10.0, collision_penalty: float = -50.0,
                 time_penalty: float = -0.1, efficiency_bonus: float = 1.0, forced_breathing: bool = True,
                 max_observed_food: int = 3, random_food_count: bool = False, respawn_food: bool = True):
        
        # Snake-specific parameters
        self.base_num_food_items = max(0, num_food_items)  # Base number for random generation
        self.random_food_count = random_food_count  # Whether to randomize food count each episode
        self.respawn_food = respawn_food  # Whether food respawns when collected
        self.food_reward = food_reward
        self.collision_penalty = collision_penalty
        self.time_penalty = time_penalty
        self.efficiency_bonus = efficiency_bonus
        self.forced_breathing = forced_breathing  # New parameter for training mode
        self.max_observed_food = max_observed_food  # Maximum food items in observation space
        
        # Current episode food count (will be set in reset)
        self.num_food_items = self.base_num_food_items
        
        # Food management
        self.food_positions = []
        self.food_radius = 15
        self.min_food_distance = 80  # Minimum distance between food items
        
        # Game state
        self.score = 0
        self.food_collected = 0
        self.steps_since_food = 0
        self.max_steps_without_food = 1500  # Increased from 500 to give more time to learn
        
        # Initialize parent class
        super().__init__(render_mode, width, height)
        
        # Set forced_breathing attribute on this environment (which IS the robot)
        # This will be used by the step method in salp_robot.py
        self.forced_breathing = forced_breathing
        
        # Update action space based on forced breathing mode
        if self.forced_breathing:
            # Single action: nozzle direction only (-1 to 1)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # Keep original two-action space: [inhale_control, nozzle_direction]
            self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # FIXED observation space for variable food counts
        # Original: [pos_x, pos_y, vel_x, vel_y, body_angle, angular_vel, body_size, breathing_phase, water_volume, nozzle_angle]
        # Added: [nearest_N_food_info (4 values each), total_food_count, avg_food_distance]
        food_obs_size = self.max_observed_food * 4 + 2  # 4 values per nearest food + 2 summary stats
        total_obs_size = 10 + food_obs_size
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -math.pi, -0.1, 0.5, 0, 0, -1] + 
                        [-1] * food_obs_size),
            high=np.array([1, 1, 10, 10, math.pi, 0.1, 2.0, 2, 1, 1] + 
                         [1] * food_obs_size),
            dtype=np.float32
        )
        
        self._generate_food_positions()
    
    def _generate_food_positions(self):
        """Generate random food positions ensuring minimum distance between them."""
        self.food_positions = []
        max_attempts = 100
        
        for _ in range(self.num_food_items):
            attempts = 0
            while attempts < max_attempts:
                # Generate random position within bounds
                x = random.uniform(self.tank_margin + self.food_radius, 
                                 self.width - self.tank_margin - self.food_radius)
                y = random.uniform(self.tank_margin + self.food_radius, 
                                 self.height - self.tank_margin - self.food_radius)
                
                # Check distance from existing food items
                valid_position = True
                for existing_pos in self.food_positions:
                    distance = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if distance < self.min_food_distance:
                        valid_position = False
                        break
                
                # Check distance from robot starting position
                robot_distance = math.sqrt((x - self.width/2)**2 + (y - self.height/2)**2)
                if robot_distance < self.min_food_distance:
                    valid_position = False
                
                if valid_position:
                    self.food_positions.append([x, y])
                    break
                
                attempts += 1
            
            # If we couldn't find a valid position, place it randomly
            if attempts >= max_attempts:
                x = random.uniform(self.tank_margin + self.food_radius, 
                                 self.width - self.tank_margin - self.food_radius)
                y = random.uniform(self.tank_margin + self.food_radius, 
                                 self.height - self.tank_margin - self.food_radius)
                self.food_positions.append([x, y])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        # Reset parent environment
        obs, info = super().reset(seed, options)
        
        # Reset snake-specific state
        self.score = 0
        self.food_collected = 0
        self.steps_since_food = 0
        
        # Set food count for this episode
        if self.random_food_count:
            # Random food count between 1 and base_num_food_items
            self.num_food_items = random.randint(1, max(1, self.base_num_food_items))
        else:
            self.num_food_items = self.base_num_food_items
        
        # Generate new food positions
        self._generate_food_positions()
        
        # Return extended observation
        extended_obs = self._get_extended_observation()
        return extended_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment forward."""
        # Step parent environment
        base_obs, base_reward, done, truncated, info = super().step(action)
        
        # Check for food collection
        food_collected = self._check_food_collection()
        
        # Check for wall collisions
        collision = self._check_wall_collision()
        
        # Calculate reward
        reward = self._calculate_snake_reward(base_reward, food_collected, collision)
        
        # Update game state
        self.steps_since_food += 1
        if food_collected:
            self.food_collected += 1
            self.score += self.food_reward
            self.steps_since_food = 0
            
            # Only respawn if respawn_food is enabled
            if self.respawn_food:
                self._respawn_food()
        
        # Check termination conditions
        if collision:
            done = True
        elif self.steps_since_food > self.max_steps_without_food:
            truncated = True
        elif not self.respawn_food and self._all_food_collected():
            # Episode ends successfully when all food collected (no respawn mode)
            done = True
        
        # Extended observation
        extended_obs = self._get_extended_observation()
        
        # Extended info
        info.update({
            'score': self.score,
            'food_collected': self.food_collected,
            'steps_since_food': self.steps_since_food,
            'collision': collision
        })
        
        return extended_obs, reward, done, truncated, info
    
    def _check_food_collection(self) -> bool:
        """Check if robot has collected any food."""
        robot_pos = self.robot_pos
        robot_radius = max(self.ellipse_a, self.ellipse_b)
        
        for i, food_pos in enumerate(self.food_positions):
            if food_pos is not None:  # Food exists
                distance = math.sqrt((robot_pos[0] - food_pos[0])**2 + 
                                   (robot_pos[1] - food_pos[1])**2)
                if distance < (robot_radius + self.food_radius):
                    # Mark food as collected
                    self.food_positions[i] = None
                    return True
        return False
    
    def _check_wall_collision(self) -> bool:
        """Check if robot has collided with walls."""
        robot_radius = max(self.ellipse_a, self.ellipse_b)
        margin = self.tank_margin
        
        # Check boundaries
        if (self.robot_pos[0] - robot_radius <= margin or 
            self.robot_pos[0] + robot_radius >= self.width - margin or
            self.robot_pos[1] - robot_radius <= margin or 
            self.robot_pos[1] + robot_radius >= self.height - margin):
            return True
        return False
    
    def _respawn_food(self):
        """Respawn a food item in a random valid location."""
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random position
            x = random.uniform(self.tank_margin + self.food_radius, 
                             self.width - self.tank_margin - self.food_radius)
            y = random.uniform(self.tank_margin + self.food_radius, 
                             self.height - self.tank_margin - self.food_radius)
            
            # Check distance from robot
            robot_distance = math.sqrt((x - self.robot_pos[0])**2 + (y - self.robot_pos[1])**2)
            if robot_distance < self.min_food_distance:
                attempts += 1
                continue
            
            # Check distance from existing food
            valid_position = True
            for existing_pos in self.food_positions:
                if existing_pos is not None:
                    distance = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if distance < self.min_food_distance:
                        valid_position = False
                        break
            
            if valid_position:
                # Find first None position and replace it
                for i, pos in enumerate(self.food_positions):
                    if pos is None:
                        self.food_positions[i] = [x, y]
                        return
            
            attempts += 1
        
        # If we couldn't find a valid position, place it randomly
        x = random.uniform(self.tank_margin + self.food_radius, 
                         self.width - self.tank_margin - self.food_radius)
        y = random.uniform(self.tank_margin + self.food_radius, 
                         self.height - self.tank_margin - self.food_radius)
        for i, pos in enumerate(self.food_positions):
            if pos is None:
                self.food_positions[i] = [x, y]
                return
    
    def _calculate_snake_reward(self, base_reward: float, food_collected: bool, collision: bool) -> float:
        """Calculate reward focused only on food collection."""
        reward = 0.0
        
        # ONLY food collection reward - no other rewards
        if food_collected:
            reward += 100.0  # Large reward for successful food collection
        
        # Collision penalty to prevent wall hitting
        if collision:
            reward -= 50.0  # Penalty for collision
        
        return reward
    
    def _all_food_collected(self) -> bool:
        """Check if all food items have been collected."""
        for food_pos in self.food_positions:
            if food_pos is not None:
                return False
        return True
    
    def _get_nearest_food_distance(self) -> Optional[float]:
        """Get distance to nearest food item."""
        min_distance = None
        robot_pos = self.robot_pos
        
        for food_pos in self.food_positions:
            if food_pos is not None:
                distance = math.sqrt((robot_pos[0] - food_pos[0])**2 + 
                                   (robot_pos[1] - food_pos[1])**2)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def _get_extended_observation(self) -> np.ndarray:
        """Get extended observation with fixed-size food information for variable food counts."""
        # Get base observation
        base_obs = super()._get_observation()
        
        # Get all valid food positions with distances
        robot_pos = self.robot_pos
        food_with_distances = []
        
        for food_pos in self.food_positions:
            if food_pos is not None:
                distance = math.sqrt((food_pos[0] - robot_pos[0])**2 + 
                                   (food_pos[1] - robot_pos[1])**2)
                food_with_distances.append((food_pos, distance))
        
        # Sort by distance (nearest first)
        food_with_distances.sort(key=lambda x: x[1])
        
        # Create fixed-size observation for nearest N food items
        food_obs = []
        
        for i in range(self.max_observed_food):
            if i < len(food_with_distances):
                food_pos, distance = food_with_distances[i]
                
                # Relative position (normalized)
                rel_x = (food_pos[0] - robot_pos[0]) / self.width
                rel_y = (food_pos[1] - robot_pos[1]) / self.height
                
                # Distance (normalized)
                norm_distance = distance / math.sqrt(self.width**2 + self.height**2)
                
                # Angle to food relative to robot orientation
                angle_to_food = math.atan2(food_pos[1] - robot_pos[1], 
                                         food_pos[0] - robot_pos[0])
                relative_angle = angle_to_food - self.robot_angle
                # Normalize angle to [-1, 1]
                while relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                while relative_angle < -math.pi:
                    relative_angle += 2 * math.pi
                norm_angle = relative_angle / math.pi
                
                food_obs.extend([rel_x, rel_y, norm_distance, norm_angle])
            else:
                # No food at this slot - fill with default values
                food_obs.extend([0, 0, 1, 0])  # Max distance, no angle
        
        # Summary statistics
        total_food_count = len(food_with_distances)
        norm_food_count = min(total_food_count / 10.0, 1.0)  # Normalize assuming max 10 food items
        
        if food_with_distances:
            avg_distance = sum(dist for _, dist in food_with_distances) / len(food_with_distances)
            norm_avg_distance = avg_distance / math.sqrt(self.width**2 + self.height**2)
        else:
            norm_avg_distance = 1.0  # Max distance when no food
        
        food_obs.extend([norm_food_count, norm_avg_distance])
        
        # Combine observations
        extended_obs = np.concatenate([base_obs, food_obs])
        return extended_obs.astype(np.float32)
    
    def render(self):
        """Render the environment with food items."""
        # Render base environment
        result = super().render()
        
        if self.render_mode is None:
            return result
        
        # Draw food items
        for food_pos in self.food_positions:
            if food_pos is not None:
                # Food item (green circle)
                pygame.draw.circle(self.screen, (50, 255, 50), 
                                 (int(food_pos[0]), int(food_pos[1])), self.food_radius)
                # Food outline
                pygame.draw.circle(self.screen, (30, 200, 30), 
                                 (int(food_pos[0]), int(food_pos[1])), self.food_radius, 2)
        
        # Draw score and game info
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            
            # Game info
            game_info = [
                f"Score: {self.score}",
                f"Food Collected: {self.food_collected}",
                f"Steps Since Food: {self.steps_since_food}",
                ""
            ]
            
            # Draw game info on the right side
            for i, line in enumerate(game_info):
                if line:  # Skip empty lines
                    text = font.render(line, True, (255, 255, 100))
                    self.screen.blit(text, (self.width - 200, 10 + i * 25))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))


def main():
    """Demo the SALP Snake environment."""
    print("SALP Snake Environment Demo")
    print("Controls:")
    print("- HOLD SPACE: Inhale water (slow contraction)")
    print("- RELEASE SPACE: Exhale water (slow expansion + thrust)")
    print("- ←/→ Arrow Keys: Steer rear nozzle left/right")
    print("- Collect green food items!")
    print("- ESC: Quit")
    
    pygame.init()
    env = SalpSnakeEnv(render_mode="human", num_food_items=5, forced_breathing=False)  # Use full control for demo
    observation, info = env.reset()
    
    running = True
    nozzle_direction = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Inhale control (hold space)
        inhale_control = 1.0 if keys[pygame.K_SPACE] else 0.0
        
        # Nozzle steering (using arrow keys) - Fixed direction like original SALP
        if keys[pygame.K_LEFT]:  # Left arrow key - steer nozzle left
            nozzle_direction = min(1.0, nozzle_direction + 0.03)
        elif keys[pygame.K_RIGHT]:  # Right arrow key - steer nozzle right
            nozzle_direction = max(-1.0, nozzle_direction - 0.03)
        # Maintain nozzle position when no input (like original SALP)
        
        # Apply action based on forced_breathing mode
        if env.forced_breathing:
            # Forced breathing mode: only nozzle control
            action = np.array([nozzle_direction])
        else:
            # Full control mode: breathing + nozzle
            action = np.array([inhale_control, nozzle_direction])
        
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if done or truncated:
            print(f"Episode ended! Score: {info['score']}, Food collected: {info['food_collected']}")
            observation, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    main()
