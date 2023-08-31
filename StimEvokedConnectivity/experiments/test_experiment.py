import numpy as np

from core.maxwell_env import MaxwellEnv
from core.params import maxwell_params
import time
import sys

params = maxwell_params
params['name'] = 'test'
params['stim_electrodes'] = [10254,14130]
params['max_time_sec'] = 60
# params['multiprocess'] = False # 998.0181806065399, 997.2547488408828, 998.715714510438
# In [2]: 1/np.mean(a)
# Out[2]: 998.9690329452344

# In [3]: np.std(a)
# Out[3]: 2.8645002508089847e-05


params['multiprocess'] = True # 995.7867787711177, 994.2808954298225
params['render'] = False
#In [2]: 1/np.mean(a)
# Out[2]: 996.1429809620076

# In [3]: np.std(a)
# Out[3]: 3.789745655358919e-05
if __name__ == '__main__':
    env = MaxwellEnv(**params)

    done = False

    neuron = 0

    q = 0
    fs = 20000
    t = time.perf_counter()
    times = []

    try:
        while not done:
            # Bind the loop to 1ms step
            dt = env.dt
            if dt >= 1/fs:
            # if time.perf_counter() - t >= 1/(fs+800):
                # q += 1
                # print (time.perf_counter() - t)
                # times.append(time.perf_counter() - t)
                times.append(dt)
                # t = time.perf_counter()
                if env.stim_dt > 1:
                #     print("Stimulating at\t",env.time_elapsed())
                    obs, done = env.step(action = ([neuron],150,100))
                    print(obs)
                #     neuron += 1
                #     neuron %= 2
                #     print("with neuron",neuron)
                # else:
                obs,done = env.step()
                # print(q)
    except KeyboardInterrupt:
        env.worker.terminate()
        env.plot_worker.terminate()
        sys.exit(1)
