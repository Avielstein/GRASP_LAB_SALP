from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
import numpy as np

def make_env():
    # Create and return the SalpRobotEnv environment
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                    max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)  # set nozzle angles
    robot.set_environment(density=1000)  # water density in kg/m^3

    env = SalpRobotEnv(render_mode=None, robot=robot)

    return env

if __name__ == "__main__":

    num_cpu = 8
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # Create separate evaluation environment
    eval_env = make_env()

    # 2. Sanity Check (CRITICAL)
    # This checks if your observation/action spaces match what the step() function returns.
    # It will crash here if you made a mistake, saving you hours of debugging.
    print("Checking environment...")
    # check_env(env)
    print("Environment is valid!")

    # 3. Define the Model (SAC)
    # V2: Continue training from v1 best model with enhanced reward function
    print("Loading v1 best model to continue training as v2...")
    model = SAC.load("../experiments/v1/models/best_model/best_model", env=vec_env)
    print("✅ v1 best model loaded successfully!")

    # 4. Setup Evaluation callback to save best model only
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='../experiments/v2/models/best_model/',
        log_path='../experiments/v2/logs/eval_logs/',
        eval_freq=1000,  # Evaluate every 1000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Run 5 episodes for evaluation
        verbose=1
    )

    # 5. Train
    print("Starting v2 training (continuing from v1 best model)...")
    print("✅ Enhanced reward: Now includes r_cycle and r_energy")
    print("✅ Eval callback: evaluates every 1000 steps, saves best model only")
    print("✅ Training for 200k additional timesteps")
    print()
    model.learn(
        total_timesteps=200000,  # 200k additional timesteps for v2
        callback=eval_callback,  # Only eval callback, no checkpoint callback
        reset_num_timesteps=False,  # Continue from v1's timestep count
        tb_log_name="salp_robot_v2"
    )

    # 6. Save Final Model
    model.save("../experiments/v2/models/salp_robot_v2_final")
    print("Training finished. Model saved as: salp_robot_v2_final")
