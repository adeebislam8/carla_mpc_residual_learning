import gymnasium as gym
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import numpy as np

from src.mpc_controller.envs.carlaEnv import CarlaMPCEnv
from racing_score_metrics import RacingMetricsTracker, RacingScoreCalculator

import tracemalloc
from tqdm import tqdm

import gc
from collections import Counter

class ObjectCountCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.baseline_counts = None
        
    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            gc.collect()
            
            # Count objects by type
            counts = Counter(type(obj).__name__ for obj in gc.get_objects())
            
            if self.baseline_counts is None:
                self.baseline_counts = counts
                print(f"📊 Baseline object counts at step {self.num_timesteps}")
            else:
                print(f"\n📊 Object count changes at step {self.num_timesteps}:")
                diffs = {k: counts[k] - self.baseline_counts.get(k, 0) 
                         for k in counts 
                         if counts[k] - self.baseline_counts.get(k, 0) > 50}
                
                for obj_type, diff in sorted(diffs.items(), key=lambda x: -x[1])[:15]:
                    print(f"  +{diff:6d}  {obj_type}  (total: {counts[obj_type]})")
        
        return True

class MemoryLeakCallback(BaseCallback):
    def __init__(self, snapshot_freq=500, top_n=20, verbose=0):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
        self.snapshots = []
        
    def _on_training_start(self):
        tracemalloc.start(25)  # 25 frames of traceback
        print("🔍 tracemalloc started")
        
    def _on_step(self):
        if self.num_timesteps % self.snapshot_freq == 0:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((self.num_timesteps, snapshot))
            
            # Compare with previous snapshot to find what GREW
            if len(self.snapshots) >= 2:
                prev_step, prev_snap = self.snapshots[-2]
                curr_step, curr_snap = self.snapshots[-1]
                
                top_stats = curr_snap.compare_to(prev_snap, 'lineno')
                
                print(f"\n📈 Memory diff: step {prev_step} → {curr_step}")
                for stat in top_stats[:10]:
                    if stat.size_diff > 0:  # Only show growing allocations
                        print(f"  +{stat.size_diff/1024:.1f} KB | {stat.count_diff} objects | {stat.traceback.format()[0]}")
            
            # Also print top absolute consumers
            top_stats_abs = snapshot.statistics('lineno')
            print(f"\n🔝 Top absolute memory consumers at step {self.num_timesteps}:")
            for stat in top_stats_abs[:5]:
                print(f"  {stat.size/1024/1024:.2f} MB | {stat.count} objects | {stat.traceback.format()[0]}")
        
        return True
    
    def _on_training_end(self):
        tracemalloc.stop()



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
        towns=['Town02'],
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
    learning_rate: float = 1e-4,
    buffer_size: int = 200000,
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
        episodes_per_town=99999999,
        target_speed=15,
        max_steps=1500
    )
    
    # Create evaluation environment
    eval_env = Monitor(CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=9999999,
        target_speed=15,
        max_steps=1500
    ))
    
    # Create SAC model
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log="./sac_mpc_tensorboard/",
        gradient_steps=1,        # Don't over-update per step
        learning_starts=5000,    # Collect more experience before training
        policy_kwargs=dict(
            net_arch=[256, 256],
            optimizer_kwargs=dict(eps=1e-5),  # Applied to all optimizers at init
        ),
        ent_coef='auto', 
        tau=0.005,                    # default, fine
        gamma=0.99,                   # fine
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./sac_mpc_rl_' + str(total_timesteps) + '_checkpoints/',
        name_prefix='sac_mpc_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./sac_mpc_rl_'+ str(total_timesteps)+ '_best/',
        log_path='./sac_mpc_rl_'+ str(total_timesteps)+'_eval/',
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
        model.save("sac_mpc_rl_"+ str(total_timesteps)+"_final")
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
    learning_rate: float = 1e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    save_freq: int = 5000,
    eval_freq: int = 5000 
):
    print("="*60)
    print("MPC + Residual RL Training with PPO")
    print("="*60)

    # Match SAC's env config
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=99999999,
        target_speed=15,
        max_steps=1500
    )

    # Match SAC's eval env config
    eval_env = Monitor(CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town01'],
        episodes_per_town=9999999,
        target_speed=15,
        max_steps=1500 
    ))

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log="./ppo_mpc_overatke_tensorboard/",
        policy_kwargs=dict(
            net_arch=[256, 256],                          # added
            optimizer_kwargs=dict(eps=1e-5),              # added
        ),
        device = "cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./ppo_mpc_overatke_'+ str(total_timesteps)+'_checkpoints/',
        name_prefix='ppo_mpc_overatke_'+ str(total_timesteps)+'_model'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./ppo_mpc_overatke_best/',
        log_path='./ppo_mpc_overatke_eval/',
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

        model.save("ppo_mpc_overatke_"+ str(total_timesteps)+"_final")
        print("\n✓ Training completed successfully")

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model.save("ppo_mpc_interrupted")

    finally:
        env.close()
        eval_env.close()


def evaluate_multi_seed(model_path: str, num_episodes: int = 100):
    SEEDS = [2547, 5555, 2910]
    
    seed_results = []
    for seed in SEEDS:
        print(f"\n{'#'*80}")
        print(f"# SEED {seed}")
        print(f"{'#'*80}")
        result = evaluate_model(model_path, num_episodes=num_episodes, seed=seed)
        seed_results.append(result)  # store the FULL result, not just summary
    
    # ── Helpers ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"MULTI-SEED AGGREGATE REPORT  ({len(SEEDS)} seeds: {SEEDS})")
    print(f"Model : {model_path}")
    print(f"{'='*80}")

    def report(label, values, fmt=".2f", suffix="", unit=""):
        mean = np.mean(values)
        std  = np.std(values)
        per_seed = "  |  ".join(
            [f"s{s}: {v:{fmt}}{suffix}" for s, v in zip(SEEDS, values)]
        )
        print(f"  {label:<32}: {mean:{fmt}}{suffix} ± {std:{fmt}}{suffix}{unit}")
        print(f"    └─ {per_seed}")

    def pool_metric(fn):
        """Apply fn to each seed's all_metrics list and return list of 3 values."""
        return [fn(r['metrics']) for r in seed_results]

    def pool_score(fn):
        return [fn(r['scores']) for r in seed_results]

    # ── SECTION 1: CARLA Standard Metrics ───────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  CARLA STANDARD METRICS")
    print(f"{'─'*80}")

    report("Driving Score (DS)",
           pool_metric(lambda ms: np.mean([m.driving_score for m in ms])),
           fmt=".2f", unit="  (0-100)")

    report("Route Completion",
           pool_metric(lambda ms: np.mean([m.route_completion * 100 for m in ms])),
           fmt=".1f", suffix="%")

    report("Success Rate",
           [r['summary']['success_rate'] * 100 for r in seed_results],
           fmt=".1f", suffix="%")

    report("Collision Rate",
           [r['summary']['collision_rate'] * 100 for r in seed_results],
           fmt=".1f", suffix="%")

    infraction_vals = []
    for r in seed_results:
        vals = [m.infractions_per_km for m in r['metrics'] if m.infractions_per_km < float('inf')]
        infraction_vals.append(np.mean(vals) if vals else 0.0)
    report("Infractions / km", infraction_vals, fmt=".2f")

    # Infraction breakdown totals per seed
    print(f"\n  Infraction Breakdown (mean totals per seed):")
    for label, attr in [
        ("Collisions (layout)",      "collisions_layout"),
        ("Collisions (vehicles)",    "collisions_vehicles"),
        ("Collisions (pedestrians)", "collisions_pedestrians"),
        ("Off-road",                 "off_road_infractions"),
        ("Route timeouts",           "route_timeouts"),
    ]:
        report(label,
               pool_metric(lambda ms, a=attr: sum(getattr(m, a) for m in ms)),
               fmt=".1f")

    # ── SECTION 2: Custom Racing Scores ─────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  CUSTOM RACING SCORES  (out of 1000)")
    print(f"{'─'*80}")

    report("Overall Score",
           pool_score(lambda ss: np.mean([s['total'] for s in ss])),
           fmt=".1f")

    report("  Median Score",
           pool_score(lambda ss: np.median([s['total'] for s in ss])),
           fmt=".1f")

    report("  Best Score",
           pool_score(lambda ss: np.max([s['total'] for s in ss])),
           fmt=".1f")

    report("  Worst Score",
           pool_score(lambda ss: np.min([s['total'] for s in ss])),
           fmt=".1f")

    print(f"\n  Score Breakdown by Category:")
    for category, max_pts in [
        ("completion",  250),
        ("speed",       200),
        ("safety",      200),
        ("smoothness",  150),
        ("precision",   150),
        ("efficiency",   50),
    ]:
        vals = pool_score(lambda ss, c=category: np.mean([s[c] for s in ss]))
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        pct    = 100 * mean_v / max_pts
        per_seed = "  |  ".join(
            [f"s{s}: {v:.1f}" for s, v in zip(SEEDS, vals)]
        )
        print(f"    {category.capitalize():<12}: {mean_v:5.1f}/{max_pts} pts "
              f"({pct:5.1f}%)  ±{std_v:.1f}")
        print(f"      └─ {per_seed}")

    # Grade distribution across all seeds pooled
    print(f"\n  Grade Distribution (pooled across all seeds):")
    from collections import Counter
    all_grades = [s['grade'] for r in seed_results for s in r['scores']]
    grade_counts = Counter(all_grades)
    total_eps = num_episodes * len(SEEDS)
    for grade in ['S', 'A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']:
        count = grade_counts.get(grade, 0)
        if count > 0:
            bar = '█' * count
            print(f"    {grade:3s} : {count:3d}/{total_eps}  ({100*count/total_eps:4.1f}%)  {bar}")

    # ── SECTION 3: Completion & Lap Time ────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  COMPLETION & LAP TIME")
    print(f"{'─'*80}")

    report("Avg Path Completion",
           pool_metric(lambda ms: np.mean([m.path_completion * 100 for m in ms])),
           fmt=".1f", suffix="%")

    # Lap time — successful episodes only
    succ_times_per_seed = [
        [m.lap_time for m in r['metrics'] if m.success] for r in seed_results
    ]
    has_success = any(len(t) > 0 for t in succ_times_per_seed)

    if has_success:
        report("Lap Time Mean (success)",
               [np.mean(t) if t else float('nan') for t in succ_times_per_seed],
               fmt=".2f", suffix="s")
        report("Lap Time Fastest",
               [np.min(t) if t else float('nan') for t in succ_times_per_seed],
               fmt=".2f", suffix="s")
        report("Lap Time Slowest",
               [np.max(t) if t else float('nan') for t in succ_times_per_seed],
               fmt=".2f", suffix="s")
        report("Lap Time Std",
               [np.std(t) if len(t) > 1 else 0.0 for t in succ_times_per_seed],
               fmt=".2f", suffix="s")
    else:
        # Fall back to all episodes
        report("Lap Time Mean (all eps, no successes)",
               pool_metric(lambda ms: np.mean([m.lap_time for m in ms])),
               fmt=".2f", suffix="s")

    # ── SECTION 4: Speed ────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  SPEED  (target: 8.33 m/s)")
    print(f"{'─'*80}")

    report("Avg Speed",
           pool_metric(lambda ms: np.mean([m.avg_speed for m in ms])),
           fmt=".2f", suffix=" m/s")

    report("Speed Consistency (std)",
           pool_metric(lambda ms: np.mean([m.speed_consistency for m in ms])),
           fmt=".2f", suffix=" m/s")

    report("Time at Target Speed",
           pool_metric(lambda ms: np.mean([m.time_at_target_speed for m in ms]) * 100),
           fmt=".1f", suffix="%")

    # ── SECTION 5: Path Following ────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  PATH FOLLOWING  (lane limit: 3.5m)")
    print(f"{'─'*80}")

    report("Avg Lateral Error",
           pool_metric(lambda ms: np.mean([m.avg_lateral_error for m in ms])),
           fmt=".3f", suffix="m")

    report("Max Lateral Error (mean)",
           pool_metric(lambda ms: np.mean([m.max_lateral_error for m in ms])),
           fmt=".3f", suffix="m")

    report("Lateral Consistency (std)",
           pool_metric(lambda ms: np.mean([m.lateral_consistency for m in ms])),
           fmt=".3f", suffix="m")

    # ── SECTION 6: Safety ───────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  SAFETY")
    print(f"{'─'*80}")

    report("Close Calls (avg/episode)",
           pool_metric(lambda ms: np.mean([m.close_calls for m in ms])),
           fmt=".1f")

    report("Lane Violations (avg/episode)",
           pool_metric(lambda ms: np.mean([m.lane_violations for m in ms])),
           fmt=".1f")

    report("Time in Danger (avg)",
           pool_metric(lambda ms: np.mean([m.time_in_danger for m in ms])),
           fmt=".2f", suffix="s")

    # ── SECTION 7: Control Smoothness ───────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  CONTROL SMOOTHNESS")
    print(f"{'─'*80}")

    report("Avg Throttle Change",
           pool_metric(lambda ms: np.mean([m.avg_throttle_change for m in ms])),
           fmt=".4f")

    report("Avg Steering Change",
           pool_metric(lambda ms: np.mean([m.avg_steering_change for m in ms])),
           fmt=".4f")

    # ── SECTION 8: Efficiency ───────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  EFFICIENCY")
    print(f"{'─'*80}")

    report("Progress / Step",
           pool_metric(lambda ms: np.mean([m.progress_per_step for m in ms])),
           fmt=".3f", suffix=" m/step")

    fuel_vals = []
    for r in seed_results:
        vals = [m.fuel_efficiency for m in r['metrics'] if m.fuel_efficiency > 0]
        fuel_vals.append(np.mean(vals) if vals else 0.0)
    report("Fuel Efficiency", fuel_vals, fmt=".2f", suffix=" m/throttle")

    # ── Footer ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Total episodes evaluated : {total_eps}  ({num_episodes} × {len(SEEDS)} seeds)")
    print(f"{'='*80}\n")

    return seed_results

def evaluate_model(model_path: str, num_episodes: int = 100, seed: int = 2547):
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
    # mpc_times_ms = []
    # ppo_times_ms = []
    model = None
    if model_path is not None:
        if 'ppo' in model_path.lower():
            model = PPO.load(model_path)
        else:
            model = SAC.load(model_path)
        print(f"Evaluating RL model: {model_path}")
    else:
        print("No model provided — evaluating MPC only")

    
    # Create environment
    env = CarlaMPCEnv(
        host='localhost',
        port=2000,
        towns=['Town02'],
        episodes_per_town=9999999,
        max_steps=1500,
        render_mode='human',
        seed=seed
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
        if ep == 0:  # record first episode only
            env.start_recording(episode_num=ep, )

        while not done:
            env.render(camera_mode='top_down')
            # Get action
            if model is not None:
                # _t_ppo_start = time.perf_counter()
                action, _states = model.predict(obs, deterministic=True)
                # _t_ppo_end = time.perf_counter()
                # ppo_times_ms.append((_t_ppo_end - _t_ppo_start) * 1000)
            else:
                action = np.array([0.0, 0.0])  # Zero residual = pure MPC
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track metrics
            idx = np.argmin(np.abs(env._road_width_s - env.current_s))
            env_state = {
                'speed': env.current_speed,
                'd': env.current_d,
                's': env.current_s,
                'throttle': env.current_throttle,
                'steering': env.current_steering,
                'obstacles': env.selected_obstacles,
                'n_min': -env._road_widths[idx, 0],
                'n_max': env._road_widths[idx, 1],
            }
            tracker.update(env_state)

            # Collect MPC timing from env
            # if hasattr(env, '_last_mpc_time_ms'):
            #     mpc_times_ms.append(env._last_mpc_time_ms)
            
            # Render
            env.render(mode='human', camera_mode='follow')
        if ep == 0:
            env.save_recording(label='hybrid', output_dir='./recordings')
            return
        
        # Finalize metrics
        metrics = tracker.finalize(env.path_length, info['done_reason'])
        scores = score_calc.calculate_score(metrics, episode_length, info['done_reason'])
        
        all_scores.append(scores)
        all_metrics.append(metrics)

        # print(f"\n{'='*60}")
        # print(f"COMPUTATIONAL TIMING REPORT")
        # print(f"{'='*60}")

        # if mpc_times_ms:
        #     mpc_mean = np.mean(mpc_times_ms)
        #     mpc_std  = np.std(mpc_times_ms)
        #     print(f"  MPCC solve time:     {mpc_mean:.2f} ± {mpc_std:.2f} ms")

        # if ppo_times_ms:
        #     ppo_mean = np.mean(ppo_times_ms)
        #     ppo_std  = np.std(ppo_times_ms)
        #     print(f"  PPO inference time:  {ppo_mean:.3f} ± {ppo_std:.3f} ms")

        # if mpc_times_ms:
        #     total_mean = np.mean(mpc_times_ms) + (np.mean(ppo_times_ms) if ppo_times_ms else 0)
        #     hz = 1000.0 / total_mean
        #     print(f"  Total per timestep:  {total_mean:.2f} ms")
        #     print(f"  Control frequency:   {hz:.1f} Hz")
        #     print(f"  Required frequency:  {1/0.05:.1f} Hz (Δt = 0.05s)")
        #     print(f"  Real-time capable:   {'✓ YES' if hz > 20 else '✗ NO'}")

        # print(f"  Samples collected:   {len(mpc_times_ms)} timesteps")
        # print(f"{'='*60}\n")
        
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
            max_steps=1500,
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
            # time.sleep(2)
        
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
    parser.add_argument('--seed',       type=int,  default=2547)
    parser.add_argument('--episodes',   type=int,  default=100)
    parser.add_argument('--multi_seed', action='store_true',
                        help='Evaluate across 3 fixed seeds and aggregate')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_environment()
    
    elif args.mode == 'train':
        if args.algo == 'sac':
            train_sac(total_timesteps=args.timesteps)
        else:
            train_ppo(total_timesteps=args.timesteps)
    
    elif args.mode == 'eval':
        if args.multi_seed:
            evaluate_multi_seed(args.model, num_episodes=args.episodes)
        else:
            evaluate_model(args.model, num_episodes=args.episodes, seed=args.seed)
    elif args.mode == 'test_mpc':
        test_mpc_only(num_steps=10000)