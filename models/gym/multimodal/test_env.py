from custom_env import AirSimDroneEnv
import numpy as np

ip_address = '127.0.0.1'
env = AirSimDroneEnv(ip_address, -12, 17, np.zeros((768,)))
