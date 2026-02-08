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
    # V4: Continue training from v3 best model
    print("Loading v3 best model to continue training as v4...")
    model = SAC.load("./logs/v3/best_model/best_model", env=vec_env)
    print("✅ v3 best model loaded successfully!")
    
    # Alternatively, start from scratch:
    # model = SAC(
    #     "MlpPolicy",           # Use standard Dense Neural Network
    #     vec_env,
    #     verbose=1,
    #     tensorboard_log="./sac_salp_robot_tensorboard/",
    #     
    #     # --- Tuning for Robotics ---
    #     learning_rate=3e-4,
    #     buffer_size=100000,    # Big memory for off-policy
    #     batch_size=512,        # Mini-batch size
    #     ent_coef='auto',       # Automatically adjust exploration (Temperature)
    #     gamma=0.99,            # Discount factor
    #     tau=0.005,             # Polyak averaging (Soft update)
    #     device="cuda" 
    # )

    # 4. Setup Saving (Checkpoints)
    # Save the model every 1000 steps so you don't lose progress if it crashes.
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./logs/v4/',
        name_prefix='salp_robot_model_v4'
    )
    
    # Setup Evaluation callback to save best model only
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/v4/best_model/',
        log_path='./logs/v4/eval_logs/',
        eval_freq=100,  # Evaluate every 100 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Run 5 episodes for evaluation
        verbose=1
    )

    # 5. Train
    print("Starting v4 training (continuing from v3 best model)...")
    print("✅ Checkpoint callback: saves every 1000 steps to ./logs/v4/")
    print("✅ Eval callback: evaluates every 100 steps, saves best model")
    print()
    model.learn(
        total_timesteps=300000, # Run for 300k more steps (v4 training)
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False,  # Continue from v3's timestep count
        tb_log_name="salp_robot_v4_run1"
    )

    # 6. Save Final Model
    model.save("salp_robot_model_v4_final")
    print("Training finished. Model saved as: salp_robot_model_v4_final")
