from unity_python_env.controller.pid import *

def get_controller(planner_name, **config):
    # if planner_name == "dwa":
    #     return DWA(**config)
    if planner_name == "pid":
        return PID(**config)
    # elif planner_name == "apf":
    #     return APF(**config)
    # elif planner_name == "rpp":
    #     return RPP(**config)
    # elif planner_name == "lqr":
    #     return LQR(**config)
    # elif planner_name == "mpc":
    #     return MPC(**config)
    else:
        raise ValueError("The `planner_name` must be set correctly.")