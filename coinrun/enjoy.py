"""
Load an agent trained with Stable Baselines3 PPO and run it in the environment.
"""
import time
import numpy as np
import glob
import os
from stable_baselines3 import PPO
from gym.envs.classic_control import rendering

from coinrun import setup_utils
from coinrun.coinrunenv import CoinRunEnv
from coinrun.config import Config
from coinrun import wrappers

def enjoy_env():
    should_render = True
    should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP

    # Create environment
    if should_eval:
        env = CoinRunEnv(
            game_type=Config.GAME_TYPE,
            num_envs=Config.NUM_ENV,
            lump_n=0,
            default_zoom=5.0,
        )
        should_render = False
    else:
        env = CoinRunEnv(
            game_type=Config.GAME_TYPE,
            num_envs=1,
            lump_n=0,
            default_zoom=5.0,
        )
        should_render = True
    
    env = wrappers.add_final_wrappers(env)
    
    # Setup visualization
    viewer = None
    if should_render:
        viewer = rendering.SimpleImageViewer()
    
    should_render_obs = not Config.IS_HIGH_RES
    
    # Load the trained model
    try:
        model = PPO.load("final_coinrun_model")
        print("Loaded model from final_coinrun_model")
    except:
        try:
            # Try to load the latest checkpoint if final model not found
            checkpoints = glob.glob("./logs/ppo_coinrun_model_*.zip")
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                model = PPO.load(latest_checkpoint)
                print(f"Loaded model from checkpoint: {latest_checkpoint}")
            else:
                raise FileNotFoundError("No model checkpoints found")
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0
    
    # Reset environment
    obs = env.reset()
    t_step = 0
    
    def maybe_render(info=None):
        if should_render and not should_render_obs:
            env.render()
    
    # Initialize tracking variables
    nenvs = env.num_envs
    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))
    
    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs
        return True
    
    # Begin evaluation loop
    maybe_render()
    done = np.zeros(nenvs, dtype=bool)
    
    while should_continue():
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, rew, done, info = env.step(action)
        
        # Render if needed
        if should_render and should_render_obs:
            if np.shape(obs)[-1] % 3 == 0:
                ob_frame = obs[0,:,:,-3:]
            else:
                ob_frame = obs[0,:,:,-1]
                ob_frame = np.stack([ob_frame] * 3, axis=2)
            viewer.imshow(ob_frame)
        
        # Update rewards
        curr_rews[:,0] += rew
        
        # Track scores for evaluation
        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1
                if 'episode' in info[i]:
                    scores[i] += info[i].get('episode')['r']
        
        # Print progress
        if t_step % 100 == 0:
            print('t', t_step, 'reward', rew[0], 'cumulative_reward', curr_rews[0], 'obs_shape', np.shape(obs))
            maybe_render(info[0] if len(info) > 0 else None)
        
        t_step += 1
        
        if should_render:
            time.sleep(0.02)
        
        if done[0]:
            if should_render:
                print('Episode reward:', curr_rews)
            curr_rews[:] = 0
    
    # Clean up
    if viewer:
        viewer.close()
    
    # Report results for evaluation mode
    result = 0
    if should_eval:
        mean_score = np.mean(scores) / rep_count
        max_idx = np.argmax(scores)
        print('Scores:', scores / rep_count)
        print('Mean score:', mean_score)
        print('Max idx:', max_idx)
        result = mean_score
    
    return result

def main():
    setup_utils.setup_and_load()
    enjoy_env()

if __name__ == '__main__':
    main()