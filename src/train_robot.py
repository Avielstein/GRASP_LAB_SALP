from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback
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


    # 2. Sanity Check (CRITICAL)
    # This checks if your observation/action spaces match what the step() function returns.
    # It will crash here if you made a mistake, saving you hours of debugging.
    # print("Checking environment...")
    # check_env(env)


    
    # Create separate evaluation environment
    eval_env = make_env()
    print("Environment is valid!")


  
    # # V2: Continue training from v1 best model with enhanced reward function
    # print("Loading v1 best model to continue training as v2...")
    # model = SAC.load("../experiments/v1/models/best_model/best_model", env=vec_env)
    # print("✅ v1 best model loaded successfully!")



    # 3. Define the Model (SAC) - V4: Continue from v3 with Fixed Rewards
    print("="*70)
    print("TRAINING VERSION 4 - Continue from v3 with Fixed Rewards")
    print("="*70)
    print("\n🔧 Reward Structure Fixed:")
    print("   - Success bonus: +10 → +500 (50x increase!)")
    print("   - Failure penalty: -5 → -200 (40x increase)")
    print("   - Timeout penalty: 0 → -50 (new)")
    print("\n📈 Expected: Success rate 10% → 60-80%")
    print("="*70 + "\n")
    
    print("Loading v3 best model to continue training as v4...")
    model = SAC.load("../experiments/v3/models/best_model/best_model", env=vec_env)
    print("✅ v3 best model loaded!")
    
    # OPTION: Train from scratch (comment out above 2 lines, uncomment below)
    # model = SAC(
    #     "MlpPolicy", vec_env, verbose=1,
    #     tensorboard_log='../experiments/v4/logs',
    #     learning_rate=3e-4, buffer_size=100000,
    #     learning_starts=1000, batch_size=256,
    #     tau=0.005, gamma=0.99,
    #     train_freq=1, gradient_steps=1, device="auto"
    # )

    # 4. Setup Callbacks with Detailed Metrics
    print("\nSetting up callbacks...")
    
    # Detailed metrics callback (logs custom metrics every 1000 steps)
    metrics_callback = DetailedMetricsCallback(log_freq=1000, verbose=1)
    
    # Evaluation callback (saves best model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='../experiments/v4/models/best_model/',
        log_path='../experiments/v4/logs/eval_logs/',
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Checkpoint callback (save model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path='../experiments/v4/models/',
        name_prefix="salp_robot_v4",
        verbose=1
    )
    
    # Combine callbacks
    callback_list = CallbackList([metrics_callback, eval_callback, checkpoint_callback])
    
    print("✅ Callbacks configured:")
    print("   - Detailed metrics logged every 1000 steps")
    print("   - Evaluation every 5000 steps")
    print("   - Checkpoints every 50000 steps")

    # 5. Train
    print("\n" + "="*70)
    print("STARTING TRAINING - 200k timesteps")
    print("="*70)
    print("📊 Monitor progress: tensorboard --logdir ../experiments/v4/logs")
    print("📖 See METRICS.md for metric documentation")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=200000,
        callback=callback_list,
        tb_log_name="salp_robot_v4",
        progress_bar=True
    )

    # 6. Save Final Model
    print("\n✅ Training complete!")
    model.save("../experiments/v4/models/salp_robot_v4_final")
    print("💾 Final model saved: salp_robot_v4_final")
    print("💾 Best model saved: ../experiments/v4/models/best_model/")
