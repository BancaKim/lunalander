"""Test official Gymnasium LunarLander-v3"""
import gymnasium as gym
import cv2
import numpy as np
from datetime import datetime

print("Testing official Gymnasium LunarLander-v3...")
print("="*60)

# Create official environment with human rendering
env = gym.make("LunarLander-v3", render_mode="human")

# Setup video recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"official_lunalander_{timestamp}.mp4"
out = None

obs, info = env.reset()
total_reward = 0
steps = 0

print("Running episode with random actions...")
print("Watch the pygame window for particle effects!")
print()

for i in range(500):  # Run for 500 steps
    # Random action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    # Try to get frame for video
    try:
        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if out is None:
                height, width = frame_bgr.shape[:2]
                out = cv2.VideoWriter(video_filename, fourcc, 50.0, (width, height))
                print(f"Recording video: {width}x{height}")
            out.write(frame_bgr)
    except:
        pass  # Some render modes don't return frames

    if (steps) % 50 == 0:
        print(f"Step {steps} | Reward: {total_reward:.2f}")

    if terminated or truncated:
        print(f"\nEpisode ended at step {steps}")
        break

env.close()

if out is not None:
    out.release()
    print(f"\n✓ Video saved: {video_filename}")

print(f"\nFinal Results:")
print(f"  Total Reward: {total_reward:.2f}")
print(f"  Total Steps: {steps}")
print("="*60)
