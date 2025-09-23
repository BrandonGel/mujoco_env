
from pettingzoo import ParallelEnv
import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Box, Dict, Graph, Discrete
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import json

from PIL import Image, ImageColor

class UnityPettingZooParallelEnv(ParallelEnv):
    metadata = {
        "name": "warehouse_parallel_v0",
    }

    def __init__(self,env_name:str):
        self.env_name = env_name
        self.config_name = 'config.json'
        unity_env = UnityEnvironment(self.env_name)
        self.env = UnityParallelEnv(unity_env)
        for agent in self.env.possible_agents:
            self.env.action_space(agent).dtype = np.dtype(np.float32)
        self.unity_action_spaces = self.env.action_spaces
        
        # Get config files
        with open(self.config_name, 'r') as f:
            self.config = json.load(f)
        with open(self.config['filepath'], 'r') as f:            
            self.param = json.load(f)
        with open(self.config['envpath'], 'r') as f:            
            self.envparam = json.load(f)
        self.img = Image.open(self.config['imagepath'])
        self.maxTimeSteps = self.param['agentParams']['maxTimeSteps']
        self.dimensions = np.array(self.envparam['map']['dimensions']) # In (x,y) world coordinate
        self.dt = self.param['unityParams']['fixed_timestep']
        self.agent_dt = self.param['unityParams']['fixed_timestep']*self.param['agentParams']['decisionPeriod']  
        self.map = np.array(self.img.resize(self.dimensions))
        binaryImage = np.dot(self.map, [0.2989, 0.5870, 0.1140])
        threshold_value = 250
        self.map[...,:] =np.where(binaryImage > threshold_value, 255, 0)[...,None]

        self.scale = np.array(self.envparam['map']['scale']) # In (x,y) world coordinate
        self.scaling = self.scale*self.dimensions # In (x,y) world coordinate

        self.start_locations = np.array(self.envparam['map']['start_locations'])
        self.goal_locations = np.array(self.envparam['map']['goal_locations'])

        # Set number of agents and tasks
        mode = self.config['mode'].lower()
        if 'generate' in mode:
            if 'paramjson' in mode:
                self.unity_num_agents = self.param['agentParams']['num_of_agents']
                self.num_tasks= self.param['goalParams']['n_tasks']
            elif 'envjson' in mode:
                self.unity_num_agents = len(self.envparam['agents'])
                self.num_tasks= self.envparam['n_tasks']
        elif 'download' in mode:
            self.unity_num_agents = len(self.envparam['agents'])
            self.num_tasks= self.envparam['n_tasks']
        self.num_envs = self.param['unityParams']['num_envs']
        self.num_starts_goals = len(self.param['goalParams']['goals']) + len(self.param['goalParams']['starts'])+1

        # Environment parameters
        self.env_id = self.env.possible_agents
        self.agent_id = [ii for ii in range(len(self.env_id))]

        angle =self.param['agentParams']['rayParams']['maxRayDegrees']
        rayDirection = self.param['agentParams']['rayParams']['rayDirections']
        if(angle == 180):
            self.num_rays = rayDirection*2
        else:
            self.num_rays = rayDirection*2+1
        self.agent_obs_dim = self.env.observation_space(self.env_id[0])[0].shape[1] - self.num_rays*3

        
        #Observation Space
        self.observation_spaces = Dict({
            env_id: Dict({
                'agent': Box(-1,1,(self.unity_num_agents,7), np.float32),
                'target': Box(-1,1,(self.unity_num_agents,5), np.float32),
                'lidar': Box(-1,1,(self.unity_num_agents,self.num_rays,3), np.float32),
                'task': Box(-1,1,self.env.observation_space(env_id)[1].shape, np.float32),
                'time': Box(0,self.maxTimeSteps ,self.env.observation_space(env_id)[2].shape, np.float32),
                'map': Box(0,255,self.map.shape, np.uint8)
            }) for env_id in self.env_id
        })
        self.action_spaces = self.env.action_spaces

    def convertToPixelCoords(self,world_coords,scale,dimensions):
        pixel_xmax, pixel_ymax = dimensions-1
        pixel_pos_flip = np.round((world_coords/scale)).astype(int)
        pixel_pos_flip[:,1]= pixel_ymax-pixel_pos_flip[:,1]
        pixel_pos = np.flip(pixel_pos_flip,axis=1)
        return pixel_pos

    def convertUnityObs(self,obs):
        observation = {}
        for agent in self.env_id:
            robotObs = obs[agent][0]
            taskObs = obs[agent][1]
            timeObs = obs[agent][2]

            robotFeat = robotObs[:, :7]
            targetFeat = robotObs[:, 7:12]
            lidarFeat = robotObs[:, 13:].reshape(self.unity_num_agents,self.num_rays,-1)
            
            agent_map = np.copy(self.map)

            agent_pixel_pos = self.convertToPixelCoords(robotFeat[:,:2],self.scale,self.dimensions)
            agent_rgb_string = "blue"
            agent_rgb = ImageColor.getrgb(agent_rgb_string)
            agent_map[agent_pixel_pos[:,0],agent_pixel_pos[:,1]] = agent_rgb

            target_ind = np.where(targetFeat[:,0] > 0)[0]
            if len(target_ind)>0:
                start_rgb_string = "green"
                start_rgb = ImageColor.getrgb(start_rgb_string)

                goal_rgb_string = "purple"
                goal_rgb = ImageColor.getrgb(goal_rgb_string)

                target_pos = targetFeat[target_ind,1:3]
                target_task_ind = targetFeat[target_ind,3].astype(int)
                target_pixel_pos = self.convertToPixelCoords(target_pos,self.scale,self.dimensions)
                
                startBool = (target_task_ind == 0) | (target_task_ind == 3)
                if np.any(startBool):
                    start_target_pixel_pos = target_pixel_pos[startBool]
                    agent_map[start_target_pixel_pos[:,0],start_target_pixel_pos[:,1]] = start_rgb
                else:
                    goal_target_pixel_pos = target_pixel_pos[~startBool]
                    agent_map[goal_target_pixel_pos[:,0],goal_target_pixel_pos[:,1]] = goal_rgb

            observation[agent] ={
                'agent': robotFeat,
                'target': targetFeat,
                'lidar': lidarFeat,
                'task': taskObs,
                'time': timeObs,
                'map':agent_map
            }
        return observation

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        observation = self.convertUnityObs(obs)
        return observation

    def step(self, actions):
        if type(actions) is not dict:
            actions = {agent: actions for agent in self.env_id}
        else:
            for ii,agent in enumerate(self.env_id):
                if agent != self.env_id[ii]:
                    print('Warning: The order of the agents is different from the environment!')
        obs, rewards, dones, infos = self.env.step(actions)
        observation = self.convertUnityObs(obs)

        rewards = {}
        for agent in self.env_id:
            rewards[agent] =obs[agent][0][:,12:13]

        return observation, rewards, dones, infos

    def render(self):
        pass

    def close(self,):
        self.env.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]