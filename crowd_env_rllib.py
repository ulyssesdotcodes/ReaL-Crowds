import time
import os
import random

import numpy as np
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # print(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        # sys.path.append(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        print("/opt/hfs18.5/houdini/python%d.%dlibs" % sys.version_info[:2])
        sys.path.insert(0, "/opt/hfs18.5/houdini/python%d.%dlibs" % sys.version_info[:2])
        print(sys.path)
        import hou
    finally:
        if hasattr(sys, "setdlopenflags"):
            sys.setdlopenflags(old_dlopen_flags)

enableHouModule()
import hou

class CrowdEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super(CrowdEnv, self).__init__()


        self.hou = hou
        hou.hipFile.load("/mnt/scratch/scratch_houdini/HoudiniProjects/ReaL-Crowds/crowd_avoid.hiplc")

        print(env_config)
        self.sim = self.hou.node(env_config['sim_path'])
        self.dop = self.sim.node("train_env")
        self.actions = self.dop.node("actions")
        self.curriculum = self.dop.node("curriculum")
        self.actions_input = self.dop.node("actions/input")
        self.observations = self.dop.node("observations/OUT")
        self.simulation = self.dop.simulation()

        self.cache = self.sim.node("recorded")

        self.dop.cook()
        geom = self.observations.geometry()

        # filter out known attributes
        self.attrib_names = self.sim.evalParm("observations").split()

        self.obs_size = 0
        for attr in geom.pointAttribs():
            if not attr.name() in self.attrib_names:
                continue
            # filter out string and dict values
            elif attr.dataType() == hou.attribData.Int or attr.dataType() == hou.attribData.Float:
                self.obs_size += attr.size()
            else:
                self.attrib_names.remove(attr.name())


        # grab actions
        self.num_actions = self.sim.evalParm("num_actions")
        self.action_dims = [self.sim.evalParm("action_choices" + str(i + 1)) for i in range(self.num_actions)]

        # setup multi-agent spaces
        self.action_space = spaces.MultiDiscrete(self.action_dims)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_size,), dtype=np.float32)
        
        self.reset()

    def step(self, action_dict):
        # step forward in the simulation
        frame = self.hou.frame() + 1
        self.hou.setFrame(frame)

        # set actions using the Add node's P.x channel
        for i, action in  action_dict.items():
            for j, ptaction in enumerate(action):
                idx = i * self.num_actions + j
                self.sim.parm("action_inputs").eval()
                self.sim.parm("action_value" + str(idx + 1)).set(ptaction.item())

        # put it over a fire
        self.simulation.dopNetNode().cook()

        geom = self.observations.geometry()

        dones = {}
        rewards = {}

        # accumulate all the dones and rewards by id attribute. This is important to keep rewards and observations tied together
        for pt in geom.points():
            id = pt.intAttribValue("id")
            done = pt.intAttribValue("done") == 1
            dones[id] = done
            if done:
                self.dones.add(id)
            else:
                self.dones.discard(id)
            rewards[id] = pt.floatAttribValue("reward")

        # dones = [dones i] for i in range(len(dones))]
        # rewards = [rewards[i] for i in range(len(rewards))]

        dones["__all__"] = self.done()

        return self.collect_obs(), rewards, dones, {}

    def set_reward_mean(self, in_phase, reward=0):
        phase = self.sim.evalParm("phase")
        if reward > self.sim.evalParm("reward_limit"):
            phase += 1
            self.sim.parm("phase").set(phase)
        if in_phase > phase:
            self.sim.parm("phase").set(in_phase)
        print(phase)
        return phase


    def reset(self):
        self.dones = set()

        self.hou.setFrame(1)
        self.sim.parm("seed").set(time.time() + random.random())
        # make sure to clear the simulation when we reset
        self.dop.parm("resimulate").pressButton()
        self.dop.cook()
        return self.collect_obs()
    
    def collect_obs(self):
        observations = {}
        geom = self.observations.geometry()

        for point in geom.points():
            ptobs = []
            for attrib_name in self.attrib_names:
                val = point.attribValue(attrib_name)
                # just make a big list
                if isinstance(val, tuple):
                    ptobs.extend(list(map(lambda x: float(x), val)))
                else:
                    ptobs.append(float(val))
            # collect observations by id to tie them to dones + rewards
            observations[point.intAttribValue("id")] = np.array(ptobs).astype(np.float32)

        return observations

    # indicates if the hou file has reached the end of the frame range
    def done(self):
        return self.hou.frame() >= self.hou.playbar.frameRange().y()

    # record the current frame using the cache node
    def record_frame(self, run_name, record):
        self.cache.parm("run_name").set(run_name)
        self.cache.parm("recording_index").set(record)
        self.cache.parm("execute").pressButton()