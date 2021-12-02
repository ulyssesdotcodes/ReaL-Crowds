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
        # print(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        # sys.path.insert(os.environ['HFS'] + "/houdini/python%d.%dlibs" % sys.version_info[:2])
        print("/opt/hfs18.5/houdini/python%d.%dlibs" % sys.version_info[:2])
        sys.path.insert(0, "/opt/hfs18.5/houdini/python%d.%dlibs" % sys.version_info[:2])
        print(sys.path)
        import hou
    finally:
        if hasattr(sys, "setdlopenflags"):
            sys.setdlopenflags(old_dlopen_flags)


import json

import gym, ray
from gym import spaces
import random
import numpy as np

from ray.rllib.agents import ppo
from ray.rllib.examples.models.shared_weights_model import \
    SharedWeightsModel1, SharedWeightsModel2, TF2SharedWeightsModel, \
    TorchSharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from crowd_env_rllib import CrowdEnv


if __name__ == "__main__":

    run_name_base = "avoid_"
    previous_runs = list(filter(lambda s: run_name_base in s, os.listdir('checkpoints')))
    if len(previous_runs) == 0:
        run_num = 1
    else:
        run_num = max([int(s.split(run_name_base)[1]) for s in previous_runs]) + 1

    run_name = run_name_base + "%02d" % run_num
    # writer = SummaryWriter(os.path.join('runs', run_name))
    # env = CrowdEnv(hou, "/obj/sim")

    # env.reset()

    # mod1 = mod2 = TorchSharedWeightsModel

    # ModelCatalog.register_custom_model("model1", mod1)
    # ModelCatalog.register_custom_model("model2", mod2)

    action_space = spaces.MultiDiscrete([3,3])
    observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)


    ray.init(address="auto")

    # import time

    # @ray.remote
    # def f():
    #     time.sleep(0.01)
    #     return ray._private.services.get_node_ip_address()

    # # Get a list of the IP addresses of the nodes that have joined the cluster.
    # set(ray.get([f.remote() for _ in range(1000)]))

    # import hou later
    enableHouModule()
    import hou


    config = ppo.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["num_gpus"] = 1
    config["num_workers"] = 2
    # config["num_cpus_per_worker"] = 8
    # config["num_gpus_per_worker"] = 0.25
    config["train_batch_size"] = 2400
    env_config = {
            # "hou": hou,
            "sim_path": "/obj/sim"
        }
    config["env_config"] = env_config

    chkpt_root = "checkpoints/" + run_name + "/"
    os.mkdir(chkpt_root)

    phase = 0


    # def train_func(config, checkpoint_dir=None):
    #     start = 0
    #     if checkpoint_dir:
    #         with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
    #             state = json.loads(f.read())
    #             start = state["step"] + 1

    #     for iter in range(start, 100):
    #         time.sleep(1)

    #         with tune.checkpoint_dir(step=step) as checkpoint_dir:
    #             path = os.path.join(checkpoint_dir, "checkpoint")
    #             with open(path, "w") as f:
    #                 f.write(json.dumps({"step": start}))

    #         tune.report(hello="world", ray="tune")

    phase = 0
    phase_file = os.getcwd() + "/" + chkpt_root + "phase.txt"
    f = open(phase_file, "w+")
    f.write("phase: " + str(phase))
    f.close()
    def reward_callback(env, reward):
        global phase

        old_phase = phase
        phase = env.set_reward_mean(phase, reward)

        if old_phase != phase:
            print("new phase" + str(phase))
            f = open(phase_file, "w+")
            f.write("phase: " + str(phase))
            f.close()

    
    def train(config, reporter, checkpoint_dir=None):
        trainer = ppo.PPOTrainer(env=CrowdEnv, config=config)
        # if checkpoint_dir:
        #     with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
        #         state = json.loads(f.read())
        #         start = state["step"] + 1

        i = 0

        while True:
            result = trainer.train()
            reporter(**result)
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: reward_callback(env, result['episode_reward_mean'])
                )
            )

            # with tune.checkpoint_dir(step=step) as checkpoint_dir:
            #     path = os.path.join(checkpoint_dir, "checkpoint")
            #     with open(path, "w") as f:
            #         f.write(json.dumps({"step": start}))

            # tune.report(hello="world", ray="tune")

            print(i)
            print(chkpt_root)
            if i % 16 == 15:
                print(pretty_print(result))
                chkpt_file = trainer.save(chkpt_root)
                print("checkpointed at ", chkpt_file)

            i += 1


    ray.tune.run(
        train,
        local_dir="./results",
        resources_per_trial={
            "cpu": 11,
            "gpu": 1
        },
        config=config,
        stop={"episode_reward_mean": 20}
    )

    # num_iters = 256


    # for i in range(num_iters):
    #     result = trainer.train()
    #     old_phase = phase
    #     trainer.workers.foreach_worker(
    #         lambda ev: ev.foreach_env(
    #             lambda env: reward_callback(env, result['episode_reward_mean'])
    #         )
    #     )


    # policy = trainer.get_policy()
    # model = policy.model

    # trainer = ppo.PPOTrainer(env=CrowdEnv, config=config)
    # trainer.restore(chkpt_file)
    # env = CrowdEnv(env_config)

    # record_count = 10

    # env.set_reward_mean(phase, 0)

    # state = env.reset()

    # for i in range(record_count):
    #     print(i)
    #     env.reset()
    #     print(run_name)
    #     env.record_frame(run_name, i)
    #     while not env.done():
    #         actions = trainer.compute_actions(state)
    #         state, rewards, dones, _ = env.step(actions)
    #         env.record_frame(run_name, i)




    # def gen_policy(i):
    #     config = {
    #         "model": {
    #             "custom_model": ["model1", "model2"][i % 2],
    #         },
    #         "gamma": random.choice([0.95, 0.99]),
    #     }
    #     return (None, observation_space, action_space, config)
    
    # policies = { "policy_0": gen_policy(0) }

    # policy_ids = list(policies.keys())

    # config = {
    #     "env": CrowdEnv,
    #     "env_config": {
    #         "num_agents": args.num_agents,
    #     },
    #     "simple_optimizer": args.simple,
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #     "num_sgd_iter": 10,
    #     "multiagent": {
    #         "policies": policies,
    #         "policy_mapping_fn": (lambda agent_id: random.choice(policy_ids)),
    #     },
    #     "framework": args.framework,
    # }
    # stop = {
    #     "episode_reward_mean": args.stop_reward,
    #     "timesteps_total": args.stop_timesteps,
    #     "training_iteration": args.stop_iters,
    # }
