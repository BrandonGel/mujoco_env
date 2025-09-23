from mujoco_env.environment.unity_petting_zoo import UnityPettingZooParallelEnv
import numpy as np
import json
import unittest 
import python_motion_planning as pmp
linux_gui_env_name = 'environment/multi_agent/Linux GUI Environment/LinuxGUIEnv.x86_64'
linux_server_env_name = 'environment/multi_agent/Linux Server Environment/LinuxServerEnv.x86_64'


env = UnityPettingZooParallelEnv(linux_gui_env_name)
# env = UnityPettingZooParallelEnv(linux_server_env_name)
obs =env.reset()


envMap = obs[ env.env_id[0]]['map']
envPMPMap = pmp.Grid(env.dimensions[0], env.dimensions[1])
obstacles = envPMPMap.obstacles
obsy,obsx = np.where(envMap.sum(axis=-1)==0)
obsy = env.dimensions[1]-obsy-1
for i in range(len(obsx)):
    obstacles.add((obsx[i],obsy[i]))

astarList = {env_id: [(0,None,0) for _ in range(env.unity_num_agents)] for env_id in env.env_id }


            
dt = env.agent_dt

for i in range(1000):
    actions = {a:env.action_space(a).sample()*0 for a in env.env_id}
    for env_id in env.env_id:
        target = obs[env_id]['target']
        for ii,assign in enumerate(target[:,0]>0):
            if not assign:
                continue
            if astarList[env_id][ii][0]==0:
                start = np.round(obs[env_id]['agent'][ii,:2]).astype(int)
                goal = np.round(target[ii,1:3]).astype(int)
                planner = pmp.AStar(start=tuple(start), goal=tuple(goal), env=envPMPMap)   # create planner
                cost, path, expand = planner.plan()  
                astarList[env_id][ii] = (cost,list(reversed(path)),0)
                # planner.plot.animation(path, str(planner), cost, expand)
            if astarList[env_id][ii][0]>0:
                path = astarList[env_id][ii][1]
                pathInd = astarList[env_id][ii][2]
                current_pos = obs[env_id]['agent'][ii,:2]
                current_angle = obs[env_id]['agent'][ii,2]
                diff = path[pathInd+1]-current_pos
                desired_angle = np.arctan2(diff[1],diff[0])*180/np.pi

                diffAngle = desired_angle - current_angle
                diffAngle = (diffAngle + 180) % 360 -180
                if(abs(diffAngle)> 5):
                    actions[env_id][3*ii+1] = -diffAngle/180
                else:
                    mag = np.linalg.norm(diff,axis=-1)
                    actions[env_id][3*ii] = 1
                    if mag < 0.1:
                        
                        astarList[env_id][ii] = (astarList[env_id][ii][0],path,pathInd+1)
                        if pathInd+1 >= len(path)-1:
                            astarList[env_id][ii] = (0,None,0)

    obs, rewards, dones, infos = env.step(actions)



env.close()
print(obs[env.env_id[0]]['target'])
        