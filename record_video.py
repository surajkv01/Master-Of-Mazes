import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from warehouse_env import WarehouseEnv

# Create environment with render_mode
env = WarehouseEnv(render_mode='rgb_array')

# Load trained model
model = PPO.load("models/ppo_warehouse")

# Run an episode and store frames
obs, _ = env.reset()
frames = []

done = False
while not done:
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

# Set up animation
fig = plt.figure()
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)

# Ensure video directory exists
video_dir = "video"
os.makedirs(video_dir, exist_ok=True)

# Save animation
ani.save(os.path.join(video_dir, "warehouse_simulation.mp4"), writer="ffmpeg", fps=3)
print("Video saved to video/warehouse_simulation.mp4")

env.close()
