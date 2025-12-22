#!/usr/bin/env python3
"""
Training script for Mars Survival RL Environment.

Uses Stable Baselines3 PPO algorithm with configurable hyperparameters.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError:
    raise ImportError(
        "stable-baselines3 required for training. "
        "Install with: pip install 'gym-mars[train]'"
    )

import gymnasium as gym

# Ensure gym_mars is registered
import gym_mars


def make_env():
    """Create a Mars Survival environment instance."""
    return gym.make("gym_mars/MarsSurvival-v0")


def train(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_freq: int = 10_000,
    eval_freq: int = 5_000,
    output_dir: str = "models",
    tensorboard_log: str = "logs",
    seed: int = 42,
):
    """
    Train a PPO agent on the Mars Survival environment.
    
    Args:
        total_timesteps: Total number of environment steps to train for
        n_envs: Number of parallel environments
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size for gradient updates
        n_epochs: Number of epochs for PPO updates
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient for exploration
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        output_dir: Directory to save models
        tensorboard_log: Directory for TensorBoard logs
        seed: Random seed
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mars_ppo_{timestamp}"
    model_dir = Path(output_dir) / run_name
    log_dir = Path(tensorboard_log) / run_name
    
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Mars Survival RL Training")
    print(f"   Model dir: {model_dir}")
    print(f"   Log dir: {log_dir}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Parallel environments: {n_envs}")
    print()
    
    # Create vectorized environments
    if n_envs > 1:
        env = make_vec_env(
            "gym_mars/MarsSurvival-v0",
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="mars_agent",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
    )
    
    print(f"📊 Model architecture:")
    print(f"   Policy: {model.policy}")
    print()
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Final model saved to: {final_path}.zip")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Mars Survival Environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps", type=int, default=100_000,
        help="Total number of environment steps"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Steps per environment per update"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Minibatch size"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="Number of PPO epochs"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--tensorboard-log", type=str, default="logs",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--save-freq", type=int, default=10_000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5_000,
        help="Evaluate every N steps"
    )
    
    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        output_dir=args.output_dir,
        tensorboard_log=args.tensorboard_log,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
