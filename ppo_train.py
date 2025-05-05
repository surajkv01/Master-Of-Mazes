import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from warehouse_env import WarehouseEnv

os.makedirs("video", exist_ok=True)


def train_agent():
    env = WarehouseEnv(grid_size=6, num_obstacles=6, num_items=2, max_steps=100)
    check_env(env, warn=True)

    log_dir = "ppo_logs"
    os.makedirs(log_dir, exist_ok=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)


    TIMESTEPS = 500_000
    model.learn(total_timesteps=TIMESTEPS)
    model.save("models/ppo_warehouse")
    print("Model training completed and saved.")

def evaluate_and_visualize(model, env, episodes=50):
    visit_counts = np.zeros((env.grid_size, env.grid_size), dtype=int)
    total_rewards = []
    total_deliveries = []

    print("\nEvaluating model performance...")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        deliveries = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Ensures action is scalar
            obs, reward, done, truncated, info = env.step(action)
            visit_counts[env.agent_pos[0], env.agent_pos[1]] += 1
            total_reward += reward
            if hasattr(env, 'just_delivered') and env.just_delivered:
                deliveries += 1

        total_rewards.append(total_reward)
        total_deliveries.append(deliveries)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Deliveries = {deliveries}")

    # Plot visit heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(visit_counts, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Visit Frequency'})
    plt.title("Agent Visit Frequency Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig("video/visit_heatmap.png")
    plt.show()

    # Plot reward progression
    plt.figure(figsize=(7, 4))
    plt.plot(total_rewards, label="Total Reward")
    plt.plot(total_deliveries, label="Deliveries", linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Evaluation Results")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("video/evaluation_plot.png")
    plt.show()

def main():
    train_agent()
    test_env = WarehouseEnv(grid_size=6, num_obstacles=6, num_items=2, max_steps=100)
    trained_model = PPO.load("models/ppo_warehouse")
    evaluate_and_visualize(trained_model, test_env)

if __name__ == "__main__":
    main()
