from fastapi import FastAPI, HTTPException, Request
import torch
import os
import gymnasium as gym
from torchrl.envs import GymWrapper, TransformedEnv, Compose, set_exploration_type, DoubleToFloat, StepCounter
import numpy as np
import uvicorn
app = FastAPI()

model_directory = 'src/models'
policy_name = 'policy_module.pth' #Speciffy your model filename here
# Full path to model file
model_path = os.path.join(model_directory, policy_name)

# Load the policy module
policy_module = torch.load(model_path, weights_only=False)



#To wrap whatever transforms and to filter only the action as output
# We highly recommend you to torch.export your model, but you can explore
# other alternatives.

base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = GymWrapper(
    gym.make("MountainCar-v0", render_mode="rgb_array"), categorical_action_encoding=  True, device = "cpu"
)

env = TransformedEnv(env, Compose(
    DoubleToFloat(),
    StepCounter(),
))

fake_td = env.base_env.fake_tensordict()
obs = fake_td['observation']

#warmup policy module
policy_module(obs)

with set_exploration_type("DETERMINISTIC"):
    exported = torch.export.export(
    policy_module.select_out_keys("action"),
    args=(),
    kwargs={'observation':obs},
    strict = False
  )

#### End of exporting

@app.get("/health")
def health():
  return {"message": "health ok"}

@app.post("/rl")
async def rl(request: Request):
  """
  Feed observation into RL model
  Returns action taken given current observation (int)
  """

  #get observation, feed into model
  input_json = await request.json()

  predictions = []

  for instance in input_json["instances"]:
    output =  exported.module()(observation=torch.tensor(instance))
    print(output)
    predictions.append({"action": output.detach().numpy().tolist()})
  return {"predictions": predictions}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
