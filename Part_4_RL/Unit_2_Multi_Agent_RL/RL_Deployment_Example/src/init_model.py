import torch.nn as nn
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
import gymnasium as gym
from torchrl.envs import GymWrapper, TransformedEnv, Compose, set_exploration_type, DoubleToFloat, StepCounter
import numpy as np
import torch

base_env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = GymWrapper(
    gym.make("MountainCar-v0", render_mode="rgb_array"), categorical_action_encoding=  True, device = "cpu"
)

env = TransformedEnv(env, Compose(
    DoubleToFloat(),
    StepCounter(),
))
num_cells = 64

# Simple Actor-Critic Setup

# You can skip these if you want, these are the underlying neural networks.
# Since we are using a Discrete policy, we need to use a Softmax to transform the outputs into action probabilities.
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(3),
    nn.Softmax(dim = -1)
)


value_net = nn.Sequential(
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(1),
)


# Actor Module
policy_module = ProbabilisticActor(
    module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    ),
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=Categorical,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

# Critic Module
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)


torch.save(policy_module, "models/policy_module.pth")