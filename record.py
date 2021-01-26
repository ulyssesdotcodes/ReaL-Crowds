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

from torch.utils.tensorboard import SummaryWriter

import crowd_env
from crowd_env import DQN

import os

def enableHouModule():
    '''Set up the environment so that "import hou" works.'''
    import sys, os

    # Importing hou will load in Houdini's libraries and initialize Houdini.
    # In turn, Houdini will load any HDK extensions written in C++.  These
    # extensions need to link against Houdini's libraries, so we need to
    # make sure that the symbols from Houdini's libraries are visible to
    # other libraries that Houdini loads.  So, we adjust Python's dlopen
    # flags before importing hou.
    if hasattr(sys, "setdlopenflags"):
        old_dlopen_flags = sys.getdlopenflags()
        # import DLFCN
        import ctypes
        sys.setdlopenflags(old_dlopen_flags | ctypes.RTLD_GLOBAL)

    try:
        import hou
    except ImportError:
        # Add $HFS/houdini/python2.7libs to sys.path so Python can find the
        # hou module.
        print(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        sys.path.append(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        print(sys.path)
        import hou
    finally:
        if hasattr(sys, "setdlopenflags"):
            sys.setdlopenflags(old_dlopen_flags)

enableHouModule()
import hou

hou.hipFile.load("/mnt/shared-WD/Development/HoudiniReel/ReaL_Crowds/integrate_pytorch.hiplc")
env = crowd_env.CrowdEnv(hou, "/obj/sim/train_env", "/obj/sim/recorded", 2)

device = torch.device("cpu")
policy_net = DQN(env.obs_size, 2, 234).to(device)
policy_net.load_state_dict(torch.load("box_move_vanilla"))
policy_net.eval()

def select_action(state):
    ret = policy_net(state)
    ret = ret.max(0)
    return ret[1].view(1,1)

def get_states():
    return list(map(lambda l: torch.from_numpy(l).to(device), env.collect_obs()))

def select_actions(states):
    return list(map(select_action, states))

record_count = 10

run_name="houdini_13"

for i in range(record_count):
    env.reset()
    states = get_states()
    env.record_frame(run_name, i)
    while not env.done():
        actions = select_actions(states)
        _, rewards, dones, _ = env.step(list(map(lambda a: a.item(), actions)))
        env.record_frame(run_name, i)
        states = get_states()