import time

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
        self.fc3 = nn.Linear(128, sum(action_size))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CrowdEnv(gym.Env):
    def create_many(hou, sim_path):
        num_agents = hou.node(sim_path).parm("num_agents")
        envs = []
        for i in range(num_agents):
            envs.append(CrowdEnv(hou, sim_path))

    def __init__(self, hou, sim_path, agent_ids):
        super(CrowdEnv, self).__init__()

        self.hou = hou

        self.sim = self.hou.node(sim_path)
        self.dop = self.sim.node("train_env")
        self.actions = self.dop.node("actions")
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
        actions_node = self.actions.node("action_options")
        self.action_dims = [int(actions_node.track(str(i)).evalAtSample(0)) for i in range(self.num_actions)]

        # setup multi-agent spaces
        self.action_space = []
        self.observation_space = []
        
        for id in range(geom.intrinsicValue("pointcount")):
            self.action_space.append(spaces.MultiDiscrete(self.action_dims))

            # automatically compute observation space
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_size,), dtype=np.float32))

        print("end init")

    def step(self, actions):
        # step forward in the simulation
        frame = self.hou.frame() + 1
        self.hou.setFrame(frame)

        # set actions using the Add node's P.x channel
        for i, action in  enumerate(actions):
            for j, ptaction in enumerate(action):
                idx = i * self.num_actions + j
                self.actions_input.parm("value" + str(idx)).set(ptaction)

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
        self.sim.parm("seed").set(time.time())
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