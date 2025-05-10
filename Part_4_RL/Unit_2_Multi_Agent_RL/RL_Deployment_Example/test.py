import requests
import gymnasium as gym
from torchrl.envs import GymWrapper, TransformedEnv, Compose, set_exploration_type, DoubleToFloat, StepCounter
import numpy as np

base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = GymWrapper(
    gym.make("MountainCar-v0", render_mode="rgb_array"), categorical_action_encoding=  True, device = "cpu"
)
# The endpoint URL
url = 'http://localhost:8000/rl'

# Example question and context
data = {
    "instances":[
        env.observation_space.sample().tolist() for i in range(3)
    ]
}

print(data)

# Sending a POST request
response = requests.post(url, json=data)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response:", response.json())
