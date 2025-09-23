from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
import numpy as np
import json
import unittest 
from PIL import Image
from pettingzoo.test import api_test

class test_unity_aec(unittest.TestCase):
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
            env = UnityAECEnv(unity_env)
            env.reset()
            for agent in env.possible_agents:
                env.action_space(agent).dtype = np.dtype(np.float32)
            agent_ids = [agent_id for agent_id in env.action_spaces.keys()]
            
            action_spaces = env.action_spaces
            for agent in env.possible_agents:
                observation, reward, done, info =  env.observe(agent)
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
                dones = []
                for jj in range(env.num_agents):
                    action =env.action_spaces[agent_ids[jj]].sample()
                    env.step(action)
                    observation, reward, done, info =  env.last()
                    dones.append(done)

                if np.any(dones):
                    env.reset()
                    ep += 1
                    print("Episode " + str(ep))
            env.close()
            print('finish')

        def test_api(name):
            unity_env = UnityEnvironment(name)
            env = UnityAECEnv(unity_env)
            api_test(env, num_cycles=self.numSteps, verbose_progress=False)
            env.close()

        if len(name)>0:
            test_env(name)
            test_api(name)

    def test_linux_env(self):
        self.test_os_env(self.linux_gui_env_name)
        self.test_os_env(self.linux_server_env_name)


if __name__ == "__main__":
    unittest.main()