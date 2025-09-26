"""
@file: pid.py
@breif: PID motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.25
"""
import numpy as np
import math

from python_motion_planning.local_planner import PID as localPID
from python_motion_planning.utils import Env, MathHelper,Grid


class PID(localPID):
    """
    Class for PID motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        **params: other parameters can be found in the parent class LocalPlanner

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import PID
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = PID(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, **params) -> None:
        super().__init__((0,0,0), (0,0,0), Grid(1,1), **params)
        # PID parameters
        self.e_w, self.i_w = 0.0, 0.0
        self.e_v, self.i_v = 0.0, 0.0
        self.actuation = np.zeros((1,2))
        if 'k_v_p' in params:
            self.k_v_p = params['k_v_p']
        if 'k_v_i' in params:
            self.k_v_i = params['k_v_i']
        if 'k_v_d' in params:
            self.k_v_d = params['k_v_d']
        if 'k_w_p' in params:
            self.k_w_p = params['k_w_p']
        if 'k_w_i' in params:
            self.k_w_i = params['k_w_i']
        if 'k_w_d' in params:
            self.k_w_d = params['k_w_d']
        if 'k_theta' in params:
            self.k_theta = params['k_theta']

    def __str__(self) -> str:
        return "PID Planner"

    def set_Robot_State(self,state:tuple):
        """
        Set the robot state.

        Parameters:
            state (tuple): robot state (x, y, theta, v, w)
        """
        self.robot.reset()
        self.robot.px = state[0]
        self.robot.py = state[1]
        self.robot.theta = state[2]
        self.robot.v = state[3]
        self.robot.w = state[4]
        self.e_w, self.i_w = 0.0, 0.0
        self.e_v, self.i_v = 0.0, 0.0

    def plan(self, state:tuple, goal:tuple = None,num_timesteps: int=1, path=None, normalize: bool = False):
        """
        PID motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """

        # Check Timesteps is set correctly
        dt = self.params["TIME_STEP"]
        assert num_timesteps > 0, "Either num_timesteps or time_duration should be set correctly."    

        # Set Goal if provided & check is Goal is set correctly
        if goal is not None:
            self.goal = goal
        assert self.goal is not None, "Goal should be set correctly."
        
        assert path is not None, "Global path should be provided for local planner."
        self.path = path

        # Set Robot State
        self.set_Robot_State(state)


        actuation = np.zeros((num_timesteps, 2))
        goal_reach = False        
        for ii in range(num_timesteps):
            # break until goal reached
            if self.reachGoal(tuple(self.robot.state.squeeze(axis=1)[0:3]), self.goal):
                goal_reach = True
                break
            
            # find next tracking point
            lookahead_pt, theta_trj, _ = self.getLookaheadPoint()

            theta_err = self.angle(self.robot.position, lookahead_pt)
            if abs(theta_err - theta_trj) > np.pi:
                if theta_err > theta_trj:
                    theta_trj += 2 * np.pi
                else:
                    theta_err += 2 * np.pi
            theta_d = self.k_theta * theta_err + (1 - self.k_theta) * theta_trj

            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2])
            if not self.shouldMoveToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(theta_d - self.robot.theta)
                if self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
                else:
                    v_d = self.dist(lookahead_pt, self.robot.position) / dt
                    u = np.array([[self.linearRegularization(v_d)], [self.angularRegularization(e_theta / dt)]])

            # feed into robotic kinematic
            self.robot.kinematic(u, dt)
            actuation[ii,:] = u.squeeze(axis=1)
        if normalize:
            actuation[:,0] = actuation[:,0]/self.params["MAX_V"]
            actuation[:,1] = actuation[:,1]/self.params["MAX_W"]
        return actuation,goal_reach