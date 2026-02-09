import gymnasium as gym
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
import numpy as np

from src.mpc_controller.envs.carlaEnv import CarlaMPCEnv


class RewardLoggerCallback(BaseCallback):
    """Log episode rewards and lap times"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def _on_step(self) -> bool:
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            
            info = self.locals['infos'][0]
            done_reason = info.get('done_reason', 'unknown')
            lap_time = info.get('lap_time', 0.0)
            
            print(f"\nEpisode finished: {done_reason}, reward={self.episode_reward:.2f}, "
                  f"length={self.episode_length}, lap_time={lap_time:.2f}s")
            
            # Reset for next episode
            self.episode_reward = 0.0
            self.episode_length = 0
        
        return True


def test_environment():
    """Test environment before training"""
    print("Testing environment...")
    
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=99999,
        max_steps=500
    )
    
    # Check if environment follows Gym interface
    check_env(env)
    print("✓ Environment check passed")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful, obs shape: {obs.shape}")
    
    # Test steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, done={done}")
        
        if done:
            obs, info = env.reset()
            print("  Episode done, reset")
    
    env.close()
    print("✓ Environment test completed\n")


def train_sac(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
    batch_size: int = 256,
    save_freq: int = 5000,
    eval_freq: int = 5000
):
    print("="*60)
    print("MPC + Residual RL Training with SAC")
    print("="*60)
    
    # Create environment
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=9999999,
        target_speed=8.33,
        max_steps=1000
    )
    
    # Create evaluation environment
    eval_env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=999999,
        max_steps=1000
    )
    
    # Create SAC model
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log="./sac_mpc_tensorboard/"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./sac_mpc_checkpoints/',
        name_prefix='sac_mpc_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./sac_mpc_best/',
        log_path='./sac_mpc_eval/',
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    reward_logger = RewardLoggerCallback()
    
    # Train
    try:
        print(f"\nStarting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, reward_logger],
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        model.save("sac_mpc_final")
        print("\n✓ Training completed successfully")
        print(f"✓ Final model saved to sac_mpc_final.zip")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model.save("sac_mpc_interrupted")
        print("✓ Model saved to sac_mpc_interrupted.zip")
    
    finally:
        env.close()
        eval_env.close()


def train_ppo(
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    save_freq: int = 5000,
    eval_freq: int = 5000
):
    """Train PPO agent (alternative to SAC)"""
    print("="*60)
    print("MPC + Residual RL Training with PPO")
    print("="*60)
    
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01', 'Town02', 'Town03', 'Town04'],
        episodes_per_town=3,
        max_steps=1000
    )
    
    eval_env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=1,
        max_steps=1000
    )
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log="./ppo_mpc_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./ppo_mpc_checkpoints/',
        name_prefix='ppo_mpc_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./ppo_mpc_best/',
        log_path='./ppo_mpc_eval/',
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    reward_logger = RewardLoggerCallback()
    
    try:
        print(f"\nStarting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, reward_logger],
            log_interval=10,
            progress_bar=True
        )
        
        model.save("ppo_mpc_final")
        print("\n✓ Training completed successfully")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model.save("ppo_mpc_interrupted")
    
    finally:
        env.close()
        eval_env.close()


def evaluate_model(model_path: str, num_episodes: int = 10):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model (without .zip extension)
        num_episodes: Number of episodes to evaluate
    """
    print(f"Evaluating model: {model_path}")
    
    # Load model
    model = SAC.load(model_path)
    
    # Create environment
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01', 'Town02', 'Town03'],
        episodes_per_town=1,
        max_steps=1000
    )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info['done_reason'] == 'success':
            success_count += 1
        
        print(f"Episode {ep+1}/{num_episodes}: reward={episode_reward:.2f}, "
              f"length={episode_length}, reason={info['done_reason']}")
    
    env.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("Evaluation Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print("="*60)

def test_mpc_only(num_steps=200, camera_mode='follow'):
    try:
        print("="*60)
        print("Testing MPC Controller (No RL)")
        print("="*60)
        
        env = CarlaMPCEnv(
            host='localhost',
            port=2000,
            towns=['Town01'],
            episodes_per_town=999999,
            max_steps=1000,
            #render_mode='human'
        )
        
        obs, info = env.reset()
        
        print(f"Path length: {env.path_length:.1f}m")
        print(f"Starting position: s={env.current_s:.1f}, d={env.current_d:.1f}")
        print("\nWatching MPC drive (no RL residuals)...\n")
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < num_steps:
            action = np.array([0.0, 0.0])
            
            try:
                obs, reward, done, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error unpacking: {e}")
                break
            env.render(camera_mode=camera_mode)
            
            total_reward += reward
            step += 1

            if step % 5 == 0:
                print(f"Step {step:3d}: "
                    f"speed={env.current_speed:4.1f} m/s, "
                    f"progress={env.current_s:5.1f}/{env.path_length:.1f}m "
                    f"({100*env.current_s/env.path_length:5.1f}%), "
                    f"lateral={env.current_d:4.2f}m, "
                    f"reward={reward:6.2f}")
            
            time.sleep(0.02)
        
        print(f"\n{'='*60}")
        print(f"Test finished!")
        print(f"  Reason: {info.get('done_reason', 'max steps')}")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final progress: {env.current_s:.1f}/{env.path_length:.1f}m "
            f"({100*env.current_s/env.path_length:.1f}%)")
        
        if 'lap_time' in info:
            print(f"  Time: {info['lap_time']:.2f}s")
        
        print(f"{'='*60}")
        
        env.close()
    except KeyboardInterrupt:
        env.close()
    except Exception as e:
        print(f"Error in test mpc only: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate MPC+RL agent')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'eval', 'test_mpc'],
                       help='Mode: train, test, eval, or test_mpc')
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'ppo'],
                       help='Algorithm: SAC or PPO')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_environment()
    
    elif args.mode == 'train':
        if args.algo == 'sac':
            train_sac(total_timesteps=args.timesteps)
        else:
            train_ppo(total_timesteps=args.timesteps)
    
    elif args.mode == 'eval':
        if args.model is None:
            print("Error: --model required for evaluation")
        else:
            evaluate_model(args.model)
    elif args.mode == 'test_mpc':
        test_mpc_only(num_steps=10000)