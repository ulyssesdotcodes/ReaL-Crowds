import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ray
from ray.rllib.agents import ppo

import crowd_env
from crowd_env_rllib import CrowdEnv

import os


ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
config["num_gpus"] = 0
config["num_workers"] = 1
env_config = {
        # "hou": hou,
        "sim_path": "/obj/sim"
    }
config["env_config"] = env_config

run_name = "avoid_88"

chkpt_root = "checkpoints/" + run_name + "/"
trainer = ppo.PPOTrainer(env=CrowdEnv, config=config)
trainer.restore("./results/train_2021-04-21_10-47-27/train_None_94d25_00000_0_2021-04-21_10-47-27/checkpoints/" + run_name + "/checkpoint_528/checkpoint-528")
env = CrowdEnv(env_config)

record_count = 10

state = env.reset()
env.set_reward_mean(4)

for i in range(record_count):
    print(i)
    env.reset()
    print(run_name)
    env.record_frame(run_name, i)
    while not env.done():
        actions = trainer.compute_actions(state)
        state, rewards, dones, _ = env.step(actions)
        env.record_frame(run_name, i)