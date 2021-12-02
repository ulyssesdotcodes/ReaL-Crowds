import numpy as np
# import gym
import crowd_env
import os
# print("reload hopefully")
# print(os.getcwd())
# reload(crowd_env)

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
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

# Instantiate the env
hou.hipFile.load("/mnt/shared-WD/Development/HoudiniReel/ReaL_Crowds/integrate_pytorch.hiplc")
env = crowd_env.CrowdEnv(hou, "/obj/sim/dopnet1")
check_env(env)
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

print("make model")
# policy = DQN('MlpPolicy', env, verbose=1)
print("learn")
# model = policy.learn(total_timesteps=10000)

# model.save("box_move")

# Test the trained agent
# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render(mode='console')
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break