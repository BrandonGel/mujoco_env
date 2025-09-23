from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import numpy as np
import json
import unittest 
from PIL import Image
from pettingzoo.test import parallel_api_test


class test_unity_parallel(unittest.TestCase):
    def setUp(self):
        self.linux_gui_env_name = 'environment/multi_agent/Linux GUI Environment/LinuxGUIEnv.x86_64'
        self.linux_server_env_name = 'environment/multi_agent/Linux Server Environment/LinuxServerEnv.x86_64'
        self.config_name = 'config.json'
        with open(self.config_name, 'r') as f:
            self.config = json.load(f)

        with open(self.config['filepath'], 'r') as f:            
            self.param = json.load(f)
        with open(self.config['envpath'], 'r') as f:            
            self.envparam = json.load(f)
        self.image = Image.open(self.config['imagepath'])
        with open(self.config['schedulepath'], 'r') as f:            
            self.schedule = json.load(f)

        mode = self.config['mode'].lower()
        if 'generate' in mode:
            if 'paramjson' in mode:
                self.num_agents = self.param['agentParams']['num_of_agents']
                self.num_tasks= self.param['goalParams']['n_tasks']
            elif 'envjson' in mode:
                self.num_agents = len(self.envparam['agents'])
                self.num_tasks= self.envparam['n_tasks']
        elif 'download' in mode:
            self.num_agents = len(self.envparam['agents'])
            self.num_tasks= self.envparam['n_tasks']
        self.num_starts_goals = len(self.param['goalParams']['goals']) + len(self.param['goalParams']['starts'])
        self.numSteps = 100

    def test_os_env(self,name = ''):  
        def test_env(name):
            unity_env = UnityEnvironment(name)
            env = UnityParallelEnv(unity_env)
            for agent in env.possible_agents:
                env.action_space(agent).dtype = np.dtype(np.float32)
            observations =  env.reset()
            action_spaces = env.action_spaces

            self.assertTrue(len(observations)>0)
            self.assertTrue(len(action_spaces)>0)

            for agent in env.possible_agents:
                observation = observations[agent]
                action_space = action_spaces[agent]

                # Action Space
                self.assertTrue(action_space.shape[0]>0)
                self.assertTrue(action_space.shape[0]==self.num_agents*3)

                # Agent Observation Space
                self.assertTrue(len(observation[0])>0)
                self.assertTrue(observation[0].shape[0]==self.num_agents)

                # Task Observation Space
                self.assertTrue(len(observation[1])>0)
                self.assertTrue(observation[1].shape[0]== self.num_tasks)
                self.assertTrue(observation[1].shape[1]==self.num_starts_goals*2+3)

                # Time Observation
                self.assertTrue(observation[2].shape[0]>0)
                self.assertTrue(observation[2].shape[0]==1)
                self.assertTrue(observation[2].shape[1]==3)
            
            ep = 1
            print("Episode " + str(ep))
            for ii in range(self.numSteps):
                actions = {a:env.action_spaces[a].sample() for a in env.action_spaces}
                observation, reward, done, info =  env.step(actions)
                if np.any([done[id] for id in env.action_spaces]):
                    observation=  env.reset()
                    ep += 1
                    print("Episode " + str(ep) + " iteration: " + str(ii))
            print('finish')
            env.close()

        def test_api(name):
            unity_env = UnityEnvironment(name)
            env = UnityParallelEnv(unity_env)
            parallel_api_test(env, num_cycles=self.numSteps)
            env.close()

        if len(name)>0:
            test_env(name)
            test_api(name)

    def test_linux_env(self):
        self.test_os_env(self.linux_gui_env_name)
        self.test_os_env(self.linux_server_env_name)

if __name__ == "__main__":
    unittest.main()






# obs_space = [env.observation_space(agent_id) for agent_id in action_spaces.keys()]
# print(obs_space)
# print(action_spaces)
# print(agent_ids)
# num_cycles=1000


# for agent in env.agent_iter(env.num_agents * num_cycles):
#     prev_observe, reward, done, info = agent.env.last()
#     if isinstance(prev_observe, dict) and 'action_mask' in prev_observe:
#         action_mask = prev_observe['action_mask']
#     if done:
#         action = None
#     else:
#         action = env.action_spaces[agent].sample() 
#     env.step(action)

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