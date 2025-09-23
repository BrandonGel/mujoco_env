from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv


# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)


# Create the channel
string_log = StringLogChannel()

# We start the communication with the Unity Editor and pass the string_log side channel as input
env_name = 'environment/multi_agent/env2.x86_64'

unity_env = UnityEnvironment(side_channels=[string_log])
# env = UnityAECEnv(unity_env)
env = UnityParallelEnv(unity_env)
string_log.send_string("The environment was reset")
env.reset()
action_spaces = env.action_spaces
print('help me')
for i in range(1000):

    # We send data to Unity : A string with the number of Agent at each
    
    actions = {a:action_spaces[a].sample() for a in action_spaces}
    observation, reward, done, info =  env.step(actions)

env.close()