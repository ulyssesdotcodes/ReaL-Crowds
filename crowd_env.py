import numpy as np
import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, obs_size, action_size, seed):
        super(DQN, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CrowdEnv(gym.Env):
    def __init__(self, hou, dop_path, cache_path, num_actions):
        super(CrowdEnv, self).__init__()
        self.hou = hou

        self.dop_path = dop_path
        self.dop = self.hou.node(dop_path)
        self.solver = self.dop.node("pytorch_solver")
        self.actions = self.solver.node("actions")
        self.simulation = self.dop.simulation()
        self.observations = self.solver.node("observations")

        self.cache = self.hou.node(cache_path)

        geom = self.observations.geometry()

        # filter out known attributes
        self.attrib_names = list(set(map(lambda a: a.name(), geom.pointAttribs())) - set(["action", "reward", "id", "done", "P"]))

        self.obs_size = 0
        for attr in geom.pointAttribs():
            if not attr.name() in self.attrib_names:
                continue
            # filter out string and dict values
            elif attr.dataType() == hou.attribData.Int or attr.dataType() == hou.attribData.Float:
                self.obs_size += attr.size()
            else:
                self.attrib_names.remove(attr.name())

        # setup multi-agent spaces
        self.action_space = []
        self.observation_space = []
        
        for id in range(geom.intrinsicValue("pointcount")):
            self.action_space.append(spaces.Discrete(num_actions))

            # automatically compute observation space
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_size,), dtype=np.float32))

        print("end init")

    def step(self, actions):
        # step forward in the simulation
        frame = self.hou.frame() + 1
        self.hou.setFrame(frame)

        # set actions using the Add node's P.x channel
        for i, action in enumerate(actions):
            self.actions.parm("usept" + str(i)).set(1)
            self.actions.parm("pt" + str(i) + "x").set(action)

        # put it over a fire
        self.simulation.dopNetNode().cook()

        geom = self.observations.geometry()

        dones = {}
        rewards = {}

        # accumulate all the dones and rewards by id attribute. This is important to keep rewards and observations tied together
        for pt in geom.points():
            id = pt.intAttribValue("id")
            dones[id] = pt.intAttribValue("done") == 1
            rewards[id] = pt.floatAttribValue("reward")

        dones = [dones[i] for i in range(len(dones))]
        rewards = [rewards[i] for i in range(len(rewards))]

        return self.collect_obs(), rewards, dones, {}


    def reset(self):
        self.hou.setFrame(1)
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

        return [observations[i] for i in range(len(observations))]

    # indicates if the hou file has reached the end of the frame range
    def done(self):
        return self.hou.frame() >= self.hou.playbar.frameRange().y()

    # record the current frame using the cache node
    def record_frame(self, run_name, record):
        self.cache.parm("run_name").set(run_name)
        self.cache.parm("recording_index").set(record)
        self.cache.parm("execute").pressButton()