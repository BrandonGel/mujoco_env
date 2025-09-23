from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from mujoco_env.environment.unity_petting_zoo import UnityPettingZooParallelEnv
import numpy as np
import json
import unittest 
import python_motion_planning as pmp
from pettingzoo.test import api_test

class test_unity_aec(unittest.TestCase):
    def setUp(self):
        self.linux_gui_env_name = 'environment/multi_agent/Linux GUI Environment/LinuxGUIEnv.x86_64'
        self.linux_server_env_name = 'environment/multi_agent/Linux Server Environment/LinuxServerEnv.x86_64'
        
    def test_env(self,):
        env = UnityPettingZooParallelEnv(self.linux_gui_env_name)
        # env = UnityPettingZooParallelEnv(self.linux_server_env_name)
        obs =env.reset()


        # envMap = obs[ env.env_id[0]]['map']
        # envPMPMap = pmp.Grid(env.dimensions[0], env.dimensions[1])
        # obstacles = envPMPMap.obstacles
        # obsy,obsx = np.where(envMap.sum(axis=-1)==0)
        # obsy = env.dimensions[1]-obsy-1
        # for i in range(len(obsx)):
        #     obstacles.add((obsx[i],obsy[i]))


        for i in range(100):
            actions = {a:env.action_space(a).sample() for a in env.env_id}
            for id in env.env_id:
                actions[id][2::3] = 0
            obs, rewards, dones, infos = env.step(actions)

        # start = np.round(obs[env.env_id[0]]['agent'][0,:2]).astype(int)
        # goal = np.round(obs[env.env_id[0]]['target'][0,1:3]).astype(int)
        # planner = pmp.AStar(start=tuple(start), goal=tuple(goal), env=envPMPMap)   # create planner
        # cost, path, expand = planner.plan()  
        # planner.plot.animation(path, str(planner), cost, expand)


        env.close()
        print(obs[env.env_id[0]]['target'])
        


if __name__ == "__main__":
    unittest.main()