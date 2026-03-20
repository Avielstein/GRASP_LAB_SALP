# test_robot.py
from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle

nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
env = SalpRobotEnv(render_mode="human", robot=robot)
# Load the trained model from the new location
# Option 1: Use the best v3 model (RECOMMENDED - best performing model)
model = SAC.load("./logs/v3/best_model/best_model", env=env)
# Option 2: Use the latest v3 checkpoint (400k steps)
# model = SAC.load("./logs/v3/salp_robot_model_v3_400000_steps", env=env)
# Option 3: Use the old v1 final model
# model = SAC.load("../data/models/v1/salp_robot_finalv2", env=env)

obs, _ = env.reset()
env.start_recording()
for i in range(100):
    # predict() returns the action and the hidden state (unused for MlpPolicy)
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)

    env.wait_for_animation()
    
    # Print or render here
    print(f"Step {i}: Action={action}, State={obs}")
    
    if truncated:
        obs, _ = env.reset()
    
    if terminated:
        print("Episode finished!")
        break
gif_path = env.stop_recording("test_robot_simulation2.gif")
env.close()
print(f"Simulation GIF saved to: {gif_path}")