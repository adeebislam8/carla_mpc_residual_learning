import gymnasium as gym
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
import numpy as np

from src.mpc_controller.envs.carlaEnv import CarlaMPCEnv
from racing_score_metrics import RacingMetricsTracker, RacingScoreCalculator


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
    print(f"{'='*80}")
    print(f"RACING EVALUATION: {model_path}")
    print(f"{'='*80}\n")
    
    # Load model
    model = SAC.load(model_path)
    
    # Create environment
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01', 'Town02', 'Town03'],
        episodes_per_town=9999999,
        max_steps=1000,
        render_mode='human'
    )
    
    score_calc = RacingScoreCalculator()
    all_scores = []
    all_metrics = []
    
    for ep in range(num_episodes):
        print(f"\n{'─'*80}")
        print(f"EPISODE {ep+1}/{num_episodes}")
        print(f"{'─'*80}")
        
        obs, info = env.reset()
        tracker = RacingMetricsTracker()
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track metrics
            env_state = {
                'speed': env.current_speed,
                'd': env.current_d,
                's': env.current_s,
                'throttle': env.current_throttle,
                'steering': env.current_steering,
                'obstacles': env.selected_obstacles
            }
            tracker.update(env_state)
            
            # Render
            env.render(mode='human', camera_mode='follow')
        
        # Finalize metrics
        metrics = tracker.finalize(env.path_length, info['done_reason'])
        scores = score_calc.calculate_score(metrics, episode_length, info['done_reason'])
        
        all_scores.append(scores)
        all_metrics.append(metrics)
        
        # Print episode summary
        print(f"\n📊 Episode {ep+1} Results:")
        print(f"  ├─ Outcome: {info['done_reason'].upper()}")
        print(f"  ├─ Lap Time: {metrics.lap_time:.2f}s")
        print(f"  ├─ Path Completion: {metrics.path_completion*100:.1f}%")
        print(f"  ├─ Avg Speed: {metrics.avg_speed:.2f} m/s")
        print(f"  ├─ Avg Lateral Error: {metrics.avg_lateral_error:.2f}m")
        print(f"  ├─ Close Calls: {metrics.close_calls}")
        print(f"  ├─ Lane Violations: {metrics.lane_violations}")
        print(f"  └─ Total Score: {scores['total']:.1f}/1000 (Grade: {scores['grade']})")
        
        print(f"\n  Score Breakdown:")
        print(f"    • Completion: {scores['completion']:.1f}/250")
        print(f"    • Speed: {scores['speed']:.1f}/200")
        print(f"    • Safety: {scores['safety']:.1f}/200")
        print(f"    • Smoothness: {scores['smoothness']:.1f}/150")
        print(f"    • Precision: {scores['precision']:.1f}/150")
        print(f"    • Efficiency: {scores['efficiency']:.1f}/50")
    
    env.close()
    
    # Final statistics
    successes = sum(1 for m in all_metrics if m.success)
    collisions = sum(1 for m in all_metrics if m.collision)
    timeouts = sum(1 for m in all_metrics if m.route_timeouts > 0)
    
    print(f"📋 CARLA STANDARD METRICS")
    print(f"{'─'*80}")
    
    # Driving Score - CARLA's official metric
    driving_scores = [m.driving_score for m in all_metrics]
    print(f"  Driving Score (DS):        {np.mean(driving_scores):6.2f} ± {np.std(driving_scores):5.2f}")
    print(f"    └─ Range: [{np.min(driving_scores):5.2f}, {np.max(driving_scores):5.2f}]")
    print(f"    └─ (Scale: 0-100, higher is better)")
    
    # Route Completion
    route_completions = [m.route_completion * 100 for m in all_metrics]
    print(f"  Route Completion (RC):     {np.mean(route_completions):6.2f}% ± {np.std(route_completions):5.2f}%")
    
    # Success Rate
    print(f"  Success Rate (SR):         {100*successes/num_episodes:6.2f}% ({successes}/{num_episodes})")
    
    # Infractions per km
    infractions_km = [m.infractions_per_km for m in all_metrics if m.infractions_per_km < float('inf')]
    if infractions_km:
        print(f"  Infractions/km:            {np.mean(infractions_km):6.2f} ± {np.std(infractions_km):5.2f}")
    else:
        print(f"  Infractions/km:            N/A (no distance traveled)")
    
    # Infraction breakdown
    print(f"\n  Infraction Breakdown (total across all episodes):")
    total_collisions_layout = sum(m.collisions_layout for m in all_metrics)
    total_collisions_vehicles = sum(m.collisions_vehicles for m in all_metrics)
    total_collisions_pedestrians = sum(m.collisions_pedestrians for m in all_metrics)
    total_off_road = sum(m.off_road_infractions for m in all_metrics)
    total_route_timeouts = sum(m.route_timeouts for m in all_metrics)
    
    print(f"    ├─ Collisions:      {total_collisions_layout}")
    print(f"    ├─ Off-road:                 {total_off_road}")
    print(f"    └─ Route timeouts:           {total_route_timeouts}")
    
    print()  # Blank line for separation
    
    print(f"🏆 CUSTOM RACING SCORES (detailed multi-dimensional analysis)")
    print(f"{'─'*80}")
    
    # Overall score
    total_scores = [s['total'] for s in all_scores]
    print(f"  Overall Score:             {np.mean(total_scores):6.1f}/1000 ± {np.std(total_scores):5.1f}")
    print(f"    ├─ Median:                 {np.median(total_scores):6.1f}")
    print(f"    ├─ Best:                   {np.max(total_scores):6.1f}")
    print(f"    ├─ Worst:                  {np.min(total_scores):6.1f}")
    
    # Grade distribution
    grades = [s['grade'] for s in all_scores]
    from collections import Counter
    grade_counts = Counter(grades)
    print(f"    └─ Grade distribution:")
    for grade in ['S', 'A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']:
        count = grade_counts.get(grade, 0)
        if count > 0:
            print(f"       └─ {grade}: {count} episodes")
    
    # Category breakdown
    print(f"\n  Score Breakdown by Category:")
    for category in ['completion', 'speed', 'safety', 'smoothness', 'precision', 'efficiency']:
        cat_scores = [s[category] for s in all_scores]
        max_score = score_calc.weights[category]
        mean_score = np.mean(cat_scores)
        percentage = 100 * mean_score / max_score
        
        print(f"    ├─ {category.capitalize():12s}: {mean_score:6.1f}/{max_score:3d} pts ({percentage:5.1f}%)")
    
    print()
    
    print(f"⚡ DETAILED PERFORMANCE METRICS")
    print(f"{'─'*80}")
    
    # Completion stats
    print(f"  Completion:")
    print(f"    ├─ Successful:             {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"    ├─ Collisions:             {collisions}/{num_episodes} ({100*collisions/num_episodes:.1f}%)")
    print(f"    ├─ Timeouts:               {timeouts}/{num_episodes} ({100*timeouts/num_episodes:.1f}%)")
    print(f"    └─ Avg completion:         {np.mean([m.path_completion for m in all_metrics])*100:.1f}%")
    
    # Lap time (only for successful episodes)
    successful_times = [m.lap_time for m in all_metrics if m.success]
    print(f"\n  Lap Time:")
    if successful_times:
        print(f"    ├─ Mean (successful):      {np.mean(successful_times):6.2f}s ± {np.std(successful_times):5.2f}s")
        print(f"    ├─ Fastest:                {np.min(successful_times):6.2f}s")
        print(f"    └─ Slowest:                {np.max(successful_times):6.2f}s")
    else:
        all_times = [m.lap_time for m in all_metrics]
        print(f"    ├─ Mean (all episodes):    {np.mean(all_times):6.2f}s ± {np.std(all_times):5.2f}s")
        print(f"    └─ ⚠️  No successful completions")
    
    # Speed metrics
    all_speeds = [m.avg_speed for m in all_metrics]
    print(f"\n  Speed:")
    print(f"    ├─ Average:                {np.mean(all_speeds):6.2f} m/s (target: 8.33 m/s)")
    print(f"    ├─ Consistency (std):      {np.mean([m.speed_consistency for m in all_metrics]):6.2f} m/s")
    print(f"    └─ Time at target:         {np.mean([m.time_at_target_speed for m in all_metrics])*100:6.1f}%")
    
    # Path following
    all_lateral_errors = [m.avg_lateral_error for m in all_metrics]
    print(f"\n  Path Following:")
    print(f"    ├─ Avg lateral error:      {np.mean(all_lateral_errors):6.3f}m (limit: 3.5m)")
    print(f"    ├─ Max lateral error:      {np.mean([m.max_lateral_error for m in all_metrics]):6.3f}m")
    print(f"    └─ Lateral consistency:    {np.mean([m.lateral_consistency for m in all_metrics]):6.3f}m")
    
    # Safety
    all_close_calls = [m.close_calls for m in all_metrics]
    all_lane_violations = [m.lane_violations for m in all_metrics]
    print(f"\n  Safety:")
    print(f"    ├─ Close calls:            {np.mean(all_close_calls):6.1f} ± {np.std(all_close_calls):5.1f}")
    print(f"    ├─ Lane violations:        {np.mean(all_lane_violations):6.1f} ± {np.std(all_lane_violations):5.1f}")
    print(f"    └─ Time in danger:         {np.mean([m.time_in_danger for m in all_metrics]):6.2f}s")
    
    # Control smoothness
    print(f"\n  Control Smoothness:")
    print(f"    ├─ Throttle changes:       {np.mean([m.avg_throttle_change for m in all_metrics]):6.4f}")
    print(f"    └─ Steering changes:       {np.mean([m.avg_steering_change for m in all_metrics]):6.4f}")
    
    # Efficiency
    all_fuel_eff = [m.fuel_efficiency for m in all_metrics if m.fuel_efficiency > 0]
    print(f"\n  Efficiency:")
    print(f"    ├─ Progress/step:          {np.mean([m.progress_per_step for m in all_metrics]):6.3f} m/step")
    if all_fuel_eff:
        print(f"    └─ Fuel efficiency:        {np.mean(all_fuel_eff):6.2f} m/throttle")
    else:
        print(f"    └─ Fuel efficiency:        N/A")
    
    print(f"\n{'─'*80}")
    print(f"🔍 DIAGNOSIS:")
    print(f"{'─'*80}")
    
    # Identify main issues
    issues = []
    
    if successes / num_episodes < 0.1:  # <10% success
        issues.append("❌ CRITICAL: Very low success rate")
        print(f"  Primary Issue: Very low success rate ({100*successes/num_episodes:.0f}%)")
        
        if collisions / num_episodes > 0.5:
            print(f"  └─ Cause: High collision rate → Focus on safety/obstacle avoidance")
        elif np.mean(all_lateral_errors) > 2.5:
            print(f"  └─ Cause: Poor lane keeping → Focus on path following")
        elif np.mean(all_speeds) < 3.0:
            print(f"  └─ Cause: Vehicle stalling → Check throttle control")
    
    elif successes / num_episodes < 0.5:  # 10-50% success
        print(f"  Status: Moderate success rate ({100*successes/num_episodes:.0f}%)")
        print(f"  └─ Improvement needed in consistency")
    
    else:  # >50% success
        print(f"  Status: Good success rate ({100*successes/num_episodes:.0f}%)")
        
        if successful_times:
            avg_time = np.mean(successful_times)
            # Rough estimate: for ~200m path at 8.33 m/s, should take ~24s
            if avg_time > 40:
                print(f"  └─ Optimization: Lap times are slow → Focus on speed")
            elif np.mean([m.avg_lateral_error for m in all_metrics if m.success]) > 1.0:
                print(f"  └─ Optimization: Path following accuracy → Focus on precision")
    
    # Specific metrics analysis
    if np.mean(all_lane_violations) > 100:
        print(f"\n  ⚠️  High lane violations ({np.mean(all_lane_violations):.0f}/episode)")
        print(f"      → Agent is frequently leaving lane boundaries")
        print(f"      → Check: lateral control gains, path following reward")
    
    if np.mean(all_close_calls) > 50:
        print(f"\n  ⚠️  Many close calls ({np.mean(all_close_calls):.0f}/episode)")
        print(f"      → Agent is driving too close to obstacles")
        print(f"      → Check: obstacle avoidance behavior, safety margins")
    
    if np.mean([m.speed_consistency for m in all_metrics]) > 3.0:
        print(f"\n  ⚠️  High speed inconsistency (std: {np.mean([m.speed_consistency for m in all_metrics]):.1f} m/s)")
        print(f"      → Agent has jerky speed control")
        print(f"      → Check: throttle smoothness penalties")
    
    print(f"\n{'='*80}\n")
    
    return {
        'scores': all_scores,
        'metrics': all_metrics,
        'summary': {
            'success_rate': successes / num_episodes,
            'collision_rate': collisions / num_episodes,
            'mean_score': np.mean(total_scores),
            'std_score': np.std(total_scores),
            'driving_score': np.mean(driving_scores),
            'infractions_per_km': np.mean(infractions_km) if infractions_km else None,
        }
    }

def test_mpc_only(num_steps=1000, camera_mode='follow'):
    env = None
    try:
        print("="*60)
        print("Testing MPC Controller (No RL)")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        env = CarlaMPCEnv(
            host='localhost',
            port=2000,
            towns=['Town01', 'Town02', 'Town03', 'Town04'],
            episodes_per_town=3,
            max_steps=1000,
            #render_mode='human'
        )
        
        episode_num = 0

        while True:
            episode_num += 1
            print(f"\n{'='*60}")
            print(f"Starting Episode {episode_num}")
            print(f"{'='*60}")
            
            obs, info = env.reset()
            
            print(f"Path length: {env.path_length:.1f}m")
            print(f"Starting position: s={env.current_s:.1f}, d={env.current_d:.1f}")
            print("\nWatching MPC drive (no RL residuals)...\n")
            
            done = False
            step = 0
            total_reward = 0
            
            # Episode loop
            while not done and step < num_steps:
                action = np.array([0.0, 0.0])
                
                try:
                    obs, reward, done, truncated, info = env.step(action)
                except Exception as e:
                    print(f"Error during step: {e}")
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
            
            # Episode summary
            print(f"\n{'='*60}")
            print(f"Episode {episode_num} finished!")
            print(f"  Reason: {info.get('done_reason', 'max steps')}")
            print(f"  Total steps: {step}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Final progress: {env.current_s:.1f}/{env.path_length:.1f}m "
                f"({100*env.current_s/env.path_length:.1f}%)")
            
            if 'lap_time' in info:
                print(f"  Time: {info['lap_time']:.2f}s")
            
            print(f"{'='*60}")
            print("Respawning in 2 seconds...")
            time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠ Testing stopped by user (Ctrl+C)")
        print(f"Total episodes completed: {episode_num}")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Error in test_mpc_only: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            print("\nClosing environment...")
            env.close()
            print("✓ Environment closed")


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