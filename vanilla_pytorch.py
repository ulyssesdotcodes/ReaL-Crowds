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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



<<<<<<< HEAD
hou.hipFile.load("/mnt/shared-WD/Development/HoudiniReel/ReaL_Crowds/integrate_pytorch.hiplc")
env = crowd_env.CrowdEnv(hou, "/obj/sim/train_env", "/obj/sim/recorded", 2)
=======
hou.hipFile.load("/mnt/shared-WD/Development/HoudiniReel/ReaL_Crowds/crowd_avoid.hiplc")
env = crowd_env.CrowdEnv(hou, "/obj/sim")
>>>>>>> ebe7570 (update nov 2021)

env.reset()

# just use cpu. we're not doing anything too complicated
device = torch.device("cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

<<<<<<< HEAD
policy_net = DQN(env.obs_size, 2, 234).to(device)
target_net = DQN(env.obs_size, 2, 234).to(device)
=======
policy_net = DQN(env.obs_size, env.action_dims, 234).to(device)
target_net = DQN(env.obs_size, env.action_dims, 234).to(device)
>>>>>>> ebe7570 (update nov 2021)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

<<<<<<< HEAD
previous_runs = os.listdir('runs')
if len(previous_runs) == 0:
  run_num = 1
else:
  run_num = max([int(s.split('houdini_')[1]) for s in previous_runs]) + 1

run_name = "houdini_%02d" % run_num
=======
previous_runs = list(filter(lambda s: "avoid_" in s, os.listdir('runs')))
if len(previous_runs) == 0:
  run_num = 1
else:
  run_num = max([int(s.split('avoid_')[1]) for s in previous_runs]) + 1

run_name = "avoid_%02d" % run_num
>>>>>>> ebe7570 (update nov 2021)
writer = SummaryWriter(os.path.join('runs', run_name))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            ret = policy_net(state)
<<<<<<< HEAD
            ret = ret.max(0)
            return ret[1].view(1,1)
    else:
        return torch.tensor([random.randrange(2)], device=device, dtype=torch.long)
=======
            # print(ret.sparse_mask(env.action_dims))
            actions = torch.tensor([])
            p = 0
            for i in env.action_dims:
                actions = torch.cat([actions,ret[range(p, p + i)].max(0)[1].view(1,1)[0]])
                p += i
            return actions.type(torch.LongTensor)
    else:
        actions = list(map(lambda h: random.randrange(h), env.action_dims))
        return torch.tensor(actions).type(torch.LongTensor)
>>>>>>> ebe7570 (update nov 2021)

def select_actions(states):
    return list(map(select_action, states))

def get_states():
    return list(map(lambda l: torch.from_numpy(l).to(device), env.collect_obs()))

running_loss = 0
avg_reward = 0
def optimize_model(current_step):
    global running_loss, avg_reward
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # batch = memory.sample(BATCH_SIZE)
    # print(batch)
    # states, actions, rewards, next_state = batch

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat(list(map(lambda x: x.unsqueeze(0), [s for s in batch.next_state
                                                if s is not None])))

    state_batch = torch.stack(batch.state)
<<<<<<< HEAD
    action_batch = torch.cat(tuple(map(lambda x: x.squeeze().unsqueeze(0).unsqueeze(1), batch.action)))
=======
    # print(batch.action)
    action_batch = torch.cat(tuple(map(lambda x: x.unsqueeze(0), batch.action)))
>>>>>>> ebe7570 (update nov 2021)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
<<<<<<< HEAD
    state_action_values = policy_net(state_batch).gather(1, action_batch)
=======
    res = policy_net(state_batch)
    state_action_values = res.gather(1, action_batch)
    # print(state_action_value)
>>>>>>> ebe7570 (update nov 2021)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
<<<<<<< HEAD
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
=======
    next_state_values = torch.zeros((BATCH_SIZE, env.num_actions), device=device)
    res = target_net(non_final_next_states)
    actions = torch.tensor([[]])
    p = 0
    for i in env.action_dims:
        print(res[:,range(p, p + i)].max(1)[0].detach().shape)
        actions = torch.cat([actions, res[:,range(p, p + i)].max(1)[0].detach()], 1)
        p += i

    next_state_values[non_final_mask] = actions.detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + torch.cat(2 * [reward_batch], 1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
>>>>>>> ebe7570 (update nov 2021)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    running_loss += loss.item()
    reward_vals = list(map(torch.mean, batch.reward))
    avg_reward += sum(reward_vals) / len(reward_vals)

    if current_step % 100 == 0:
        writer.add_scalar('training_loss', running_loss / 100, current_step)
        writer.add_scalar('avg reward', avg_reward, current_step)
        running_loss = 0
        avg_reward = 0

<<<<<<< HEAD
num_episodes = 50
=======
num_episodes = 200
>>>>>>> ebe7570 (update nov 2021)
steps = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    states = get_states()
    times = [0 for _ in states]
    for t in count():
        # Select and perform an action
        actions = select_actions(states)
<<<<<<< HEAD
        _, rewards, dones, _ = env.step(list(map(lambda a: a.item(), actions)))
=======
        # print(actions)
        _, rewards, dones, _ = env.step(list(map(lambda a: a.tolist(), actions)))
>>>>>>> ebe7570 (update nov 2021)
        rewards = list(map(lambda r: torch.tensor([r], device=device), rewards))

        # Observe new state
        next_states = get_states()

        # Store the transition in memory
        for state, action, next_state, reward in zip(states, actions, next_states, rewards):
            memory.push(state, action, next_state, reward)

        # Move to the next state
        states = next_states

        # Perform one step of the optimization (on the target network)
        optimize_model(steps)
        for i, done in enumerate(dones):
            if done:
                episode_durations.append(times[i] + 1)
                times[i] = 0
                # Update the target network, copying all weights and biases in DQN
                if i_episode % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        if env.done():
            break
        
        times = [x + 1 for x in times]
        steps += 1


# env.render()

torch.save(policy_net.state_dict(), "box_move_vanilla")


record_count = 10

for i in range(record_count):
    env.reset()
    states = get_states()
    env.record_frame(run_name, i)
    while not env.done():
        actions = select_actions(states)
        _, rewards, dones, _ = env.step(list(map(lambda a: a.tolist(), actions)))
        env.record_frame(run_name, i)
        states = get_states()