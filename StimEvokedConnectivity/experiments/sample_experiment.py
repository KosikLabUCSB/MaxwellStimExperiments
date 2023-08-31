import numpy as np

from core.maxwell_env import MaxwellEnv
from core.params import maxwell_params
import time
import sys


params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60

params['multiprocess'] = True
params['render'] = True


if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    t = time.perf_counter()
    times = []

    while not done:
        # Bind the loop to 1ms step
        if env.dt >= 1/fs:
            if env.stim_dt > 1:
                print("Stimulating at\t",env.time_elapsed())
                env.step(action = ([neuron],150,100))
                neuron ^= 1
            else:
                obs,done = env.step()
            