"""
Train an agent using Stable Baselines3 PPO instead of manual PPO implementation.
"""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

import coinrun.main_utils as utils
from coinrun import setup_utils
from coinrun.config import Config

def make_env(seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = utils.make_general_env(1, seed=seed)[0]
        return env
    return _init

def main():
    args = setup_utils.setup_and_load()
    
    # Set seed for reproducibility
    seed = int(time.time()) % 10000
    set_random_seed(seed)
    
    # Create vectorized environments
    nenvs = Config.NUM_ENVS
    env_fns = [make_env(seed + i) for i in range(nenvs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)  # Adds episode stats monitoring
    
    # Prepare model save callback
    save_interval = args.save_interval
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path="./logs/",
        name_prefix="ppo_coinrun_model"
    )
    
    # Set up model parameters
    total_timesteps = int(256e6)
    
    # Create and train the model
    model = PPO(
        "CnnPolicy",  # Default CNN policy (adjust if needed for your custom policy)
        env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.NUM_STEPS,
        batch_size=Config.NUM_STEPS * nenvs // Config.NUM_MINIBATCHES,
        n_epochs=Config.PPO_EPOCHS,
        gamma=Config.GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=Config.ENTROPY_COEFF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/tensorboard/",
        verbose=1,
        seed=seed
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save final model
    model.save("final_coinrun_model")

if __name__ == '__main__':
    main()