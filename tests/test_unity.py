from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import numpy as np
import time

# env_name = '/home/brandon-ho/Documents/code/mujoco_env/environment/single_agent/env.x86_64'
env_name = '/home/brandon-ho/Documents/code/mujoco_env/environment/multi_agent/env2.x86_64'
# env_name = '/home/brandon-ho/Documents/code/mujoco_env/environment/multi_agent/env3.x86_64'
print('help me')
unity_env = UnityEnvironment(env_name)
print('help me again')
# env = UnityAECEnv(unity_env)
env = UnityParallelEnv(unity_env)
print('help me again please')
for agent in env.possible_agents:
    env.action_space(agent).dtype = np.float32
observation=  env.reset()
obs_space = env.observation_space
action_spaces = env.action_spaces

agent_ids = [agent_id for agent_id in action_spaces.keys()]
print(obs_space)
print(action_spaces)
print(agent_ids)
num_cycles=1000
# for agent in env.agent_iter(env.num_agents * num_cycles):
#     prev_observe, reward, done, info = env.last()
#     if isinstance(prev_observe, dict) and 'action_mask' in prev_observe:
#         action_mask = prev_observe['action_mask']
#     if done:
#         action = None
#     else:
#         action = env.action_spaces[agent].sample() 
#     env.step(action)
# print(observation['Robot?team=0?agent_id=0'][0].shape)
# print(observation['Robot?team=0?agent_id=0'][0])
action_len = action_spaces[agent_ids[0]].shape
for ii in range(num_cycles):
    actions = {a:np.random.uniform(low=-1.0, high=1.0, size=action_spaces[a].shape) for a in action_spaces}
    for a in actions:
        pass
        # action = actions[a].reshape((-1,3))
        # action[:,:2] = np.random.uniform(low=-1.0, high=1.0, size=action[:,:2].shape)
        # action[:,1] = 0
        # actions[a][0::3] = 0
        # actions[a][1::3] = 0
        # actions[a][2::3] = 0
        # actions[a] = action.flatten()
    observation, reward, done, info =  env.step(actions)
    # print(actions)
env.close()
# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     action = policy(observation, agent)
#     env.step(action)


# unity_env = UnityEnvironment(
#         file_name=env_name,
#         worker_id=13,
#         no_graphics=True,
# )
# # print(unity_env)
# env = UnityToGymWrapper(unity_env, uint8_visual=False)
# print(env)
# print(env.observation_space)
# print(env.action_space)
# env.close()

# import ray 
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.tune.registry import register_env
# from tqdm import tqdm
# worker_num = 1
# def env_creator(env_config):
#     unity_env = UnityEnvironment(
#             env_name,
#             no_graphics=True,
#             worker_id=15,
#     )
#     env = UnityToGymWrapper(unity_env, uint8_visual=False)
#     return env 

# register_env('my-env', env_creator)

# ray.init() 

# config = (
#     PPOConfig()
#     .environment(env='my-env')
#     # .rollouts(num_rollout_workers=worker_num, 
#     #           num_envs_per_worker=1)
#     # .framework('torch')
#     .training(
#         model={'fcnet_hiddens': [512, 512],
#                'vf_share_layers': True,
#               },
#         gamma=0.999,
#         lr=1e-5,
#     )
#     # .evaluation(evaluation_num_workers=1, 
#     #             evaluation_duration=10000)
#     # .resources(num_gpus=int(2), 
#     #         num_cpus_per_worker=1, 
#     #         num_gpus_per_worker=0)  
#     # .reporting(min_sample_timesteps_per_iteration=1000)
#     # .debugging(log_level='ERROR') # INFO, DEBUG, ERROR, WARN
    
# )

# algo = config.build_algo()

# iteration_num = 1000
# for iteration in tqdm(range(iteration_num)):
#     result = algo.train()