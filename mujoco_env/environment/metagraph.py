import numpy as np

class metagraph():
    def __init__(self,):
        self.node_feats= {}
        self.edge_feats ={}
        self.edge_links = {}


    def init_nodes(self,):
        self.nodes = [
            'agent',
            'obstacle',
            'target',
            'reward',
            'task',
            'time'
        ]

    def init_edges(self,):
        self.edges = [
            ('obstacle', 'lidar' ,'agent'),
            ('target', 'reach' ,'agent'),
            ('reward', 'connect' ,'agent'),
            ('agent', 'assign' ,'task'),
        ]

    def build_agents(self,agent_obs):
        agent_feat_shape = (agent_obs.shape[0]+1, agent_obs.shape[1])
        agent_obs_pad = np.zeros(agent_feat_shape)
        agent_obs_pad[1:,:] = agent_obs
        self.node_feats['agent']  = agent_obs_pad
    
    def build_targets(self,target_obs):
        target_feat = np.unique(target_obs,axis=0)
        target_feat = np.sort(target_feat,axis=0)
        agent_id = np.arange(target_obs.shape[0])+1

        self.node_feats['target']  = target_feat
        self.edge_links['target', 'reach' ,'agent'] = (target_obs[:,0], agent_id)
    
    def build_rewards(self,reward_obs):
        reward_id = np.arange(reward_obs.shape[0])+1
        agent_id = np.arange(reward_obs.shape[0])+1

        self.node_feats['reward']  = reward_obs
        self.edge_links['reward', 'connect' ,'agent'] = (reward_id, agent_id)

    def build_obstacles(self,lidar_obs):
        num_agents = lidar_obs.shape[0]
        num_lidar_per_agents = lidar_obs.shape[1]//3

        agent_id = np.arange(num_agents).reshape(-1,1).repeat(num_lidar_per_agents,axis=1).flatten()+1
        obs_id = np.arange(num_agents*num_lidar_per_agents)
        lidar_obs_all = lidar_obs.reshape(-1,3)
        mask = lidar_obs_all[:,-1] ==1
        obs_feat = lidar_obs_all[mask]
        agent_id_valid = agent_id[mask]
        obs_id_valid = obs_id[mask]

        self.node_feats['obstacle']  = obs_feat
        self.edge_links['obstacle', 'lidar' ,'agent'] = (obs_id_valid, agent_id_valid)

    def build_tasks(self,task_obs):
        self.node_feats['task']  = task_obs
        task_id = np.arange(task_obs.shape[0])
        agent_id = task_obs[:,-2]
        self.edge_links['task', 'assign' ,'agent'] = (task_id, agent_id)

    def build_graph(self, observation):
        agent_obs = observation[0][:,:7]
        self.build_agents(agent_obs)

        target_obs = observation[0][:,7:11]
        self.build_targets(target_obs)

        reward_obs = observation[0][:,12]
        self.build_rewards(reward_obs)
        
        lidar_obs = observation[0][:,13:]
        self.build_obstacles(lidar_obs)
        
        return self.node_feats, self.edge_feats,self.edge_links