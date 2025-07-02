"""
Robot Deployment Script for PyTorch PPO Models

Inspired by CleanRL's evaluate function but adapted for robot deployment.
This script loads a trained PPO model and runs it on the Pupper robot or in simulation.

Usage:
python puppersim/pupper_deploy_ppo.py --model_path runs/model.cleanrl_model
"""

import os
import time
import numpy as np
import torch
import gymnasium as gym
import pickle
import argparse

# Import existing classes to avoid duplication
from pupper_train_ppo_cont_action import Agent

def make_deployment_env(env_id, run_on_robot=False, render=True, gamma=0.99):
    """
    Create environment with same wrappers as training.
    Uses the existing PupperGymEnv class.
    """
    def thunk():
        # Use the existing PupperGymEnv class with proper rendering setup
        if render and not run_on_robot:
            env = gym.make(env_id, render_mode="human", render=True)
        else:
            env = gym.make(env_id, render_mode=None, render=False)
        
        # Apply same wrappers as training for consistency
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        
        # Observation normalization
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        
        return env
    return thunk


def deploy_policy(model_path, run_on_robot=False, num_episodes=10, max_steps=10000, 
                 device='cpu', render=True, log_to_file=False, env_id="PupperGymEnvLong-v0"):
    """
    Deploy a trained PPO policy on robot or in simulation.
    
    Args:
        model_path: Path to the .cleanrl_model file
        run_on_robot: Whether to run on real robot
        num_episodes: Number of episodes to run (ignored if run_on_robot=True)
        max_steps: Maximum steps per episode
        device: PyTorch device
        render: Whether to render (simulation only)
        log_to_file: Whether to log data to file
        env_id: Name of the environment
    """
    
    # Set up device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env_fn = make_deployment_env(env_id=env_id, run_on_robot=run_on_robot, render=render)
    env = env_fn()
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    # Create a mock envs object for Agent initialization (matching training script)
    mock_envs = type('MockEnvs', (), {
        'single_observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        'single_action_space': gym.spaces.Box(low=-1, high=1, shape=(action_dim,))
    })()
    
    agent = Agent(mock_envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    print("Model loaded successfully!")
    
    # Deployment loop
    episode_returns = []
    episode_lengths = []
    
    # Logging setup
    log_dict = {
        't': [],
        'IMU': [],
        'MotorAngle': [],
        'action': [],
        'obs': [],
        'reward': []
    } if log_to_file else None
    
    try:
        # Run episodes (or continuous for robot)
        episodes_to_run = 1 if run_on_robot else num_episodes
        
        for episode in range(episodes_to_run):
            print(f"\n--- Episode {episode + 1}/{episodes_to_run} ---")
            
            # Reset environment
            try:
                # Try new gym API first
                obs, info = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # Handle new gym API
            except TypeError:
                # Fall back to old gym API (no seed argument)
                obs = env.reset()
                info = {}
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            # Timing setup
            start_time = time.time()
            
            # Episode loop
            while not done or run_on_robot:
                
                # Get action from policy
                policy_start = time.time()
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action_tensor = agent.actor_mean(obs_tensor)
                    action = action_tensor.cpu().numpy().flatten()
                
                # Step environment
                step_start = time.time()
                step_result = env.step(action)
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                
                # Update episode stats
                episode_return += reward
                episode_length += 1
                obs = next_obs
                
                # Optional manual rendering for better visualization control
                if render and not run_on_robot and hasattr(env, 'render'):
                    try:
                        env.render()
                    except:
                        pass  # Ignore rendering errors
                
                # Logging
                if log_to_file and log_dict:
                    if hasattr(env, 'robot'):
                        log_dict['t'].append(env.robot.GetTimeSinceReset())
                    else:
                        log_dict['t'].append(episode_length * 0.01)  # Approximate
                    
                    # Log motor angles and IMU if available
                    if len(obs) >= 16:
                        log_dict['MotorAngle'].append(obs[0:12].copy())
                        log_dict['IMU'].append(obs[12:16].copy())
                    else:
                        log_dict['MotorAngle'].append(obs[:min(12, len(obs))].copy())
                        log_dict['IMU'].append(obs[min(12, len(obs)):min(16, len(obs))].copy())
                    
                    log_dict['action'].append(action.copy())
                    log_dict['obs'].append(obs.copy())
                    log_dict['reward'].append(reward)
                  
                # Safety check for max steps
                if episode_length >= max_steps:
                    print(f"Episode terminated: reached max steps ({max_steps})")
                    break
                
                # Print periodic updates
                if episode_length % 100 == 0:
                    print(f"Step {episode_length}, Return: {episode_return:.2f}")
            
            # Episode summary
            episode_duration = time.time() - start_time
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {episode_length}")
            print(f"  Return: {episode_return:.2f}")
            print(f"  Duration: {episode_duration:.2f}s")
            print(f"  Avg step time: {episode_duration/episode_length:.4f}s")
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save logs
        if log_to_file and log_dict:
            log_filename = f"pupper_ppo_deployment_log_{int(time.time())}.pkl"
            print(f"Saving log to {log_filename}")
            with open(log_filename, "wb") as f:
                pickle.dump(log_dict, f)
        
        # Close environment
        env.close()
    
    # Results summary
    if episode_returns:
        print(f"\n--- Deployment Summary ---")
        print(f"Episodes completed: {len(episode_returns)}")
        print(f"Mean return: {np.mean(episode_returns):.2f}")
        print(f"Std return: {np.std(episode_returns):.2f}")
        print(f"Min return: {np.min(episode_returns):.2f}")
        print(f"Max return: {np.max(episode_returns):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    
    return episode_returns, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO policy to Pupper robot")
    
    # Model and deployment settings
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the .cleanrl_model file')
    parser.add_argument('--run_on_robot', action='store_true',
                       help='Deploy on real robot (default: simulation)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run model on')
    parser.add_argument('--env_id', type=str, default="PupperGymEnvLong-v0",
                       help='Environment ID')
    
    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes (ignored for robot)')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum steps per episode')
    
    # Visualization and logging
    parser.add_argument('--render', action='store_true', 
                       help='Render simulation (ignored for robot)')
    parser.add_argument('--no-render', dest='render', action='store_false',
                       help='Disable rendering (default: render enabled)')
    parser.set_defaults(render=True)  # Default to rendering enabled
    parser.add_argument('--log_to_file', action='store_true',
                       help='Log data to file')
    
    args = parser.parse_args()
    
    print("=== Pupper PPO Policy Deployment ===")
    print(f"Model: {args.model_path}")
    print(f"Target: {'Real Robot' if args.run_on_robot else 'Simulation'}")
    print(f"Device: {args.device}")
    
    # Deploy policy
    episode_returns, episode_lengths = deploy_policy(
        model_path=args.model_path,
        run_on_robot=args.run_on_robot,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
        env_id=args.env_id,
        render=args.render,
        log_to_file=args.log_to_file
    )


if __name__ == "__main__":
    main()
