import numpy as np

from core.maxwell_env import MaxwellEnv
from core.params import maxwell_params
import time
import sys

''' Ray: edits
 full_electrode_list = [...]
 num_repeats = math.ceil(len(full_electrode_list)/32)
 num_last_repeat = len(full_electrode_list)%32
'''

params = maxwell_params
params['name'] = 'causal_freq' # Name of the experiment
params['stim_electrodes'] = [10254,14130, 2, 4] # Define the 4 electrodes to stimulate here
params['max_time_sec'] = 60*60*2 # 2 hours
params['save_dir'] = 'data' # Path to the data directory, will be created if it doesn't exist
params['config'] = 'config.cfg' # Path to the config file

# params['multiprocess'] = False
# params['render'] = False

# Lets define the stim_cycle in seconds
tetanus_Hz = 50
causal_Hz = 1

stim_cycle = [
    ('tetanus', 60),
    ('silent', 60),
    ('causal', 120),
    ('silent', 60)
]
silent_cycle = [
    ('silent', 60),
    ('silent', 10),
    ('causal', 120),
    ('silent', 60)
]

full_exp = 3*silent_cycle + 5*stim_cycle + 1*silent_cycle + 6*stim_cycle + 1*silent_cycle + 5*stim_cycle + 3*silent_cycle

delay_ms = 5
# We will do this at tetanus_Hz
tetanus_action = [('stim',[0],150,100), ('delay', delay_ms), 
                  ('stim',[1],150,100), ('delay', delay_ms),
                  ('stim',[2],150,100), ('delay', delay_ms),
                  ('stim',[3],150,100)]

''' Ray: tetanus_action2 = [('stim',[0],150,100), ('delay', delay_ms), 
                  ('stim',[1],150,100), ('delay', delay_ms),
                  ('stim',[2],150,100), ('delay', delay_ms),
                  ('stim',[3],150,100), ('delay', delay_ms), 
                  ('stim',[4],150,100), ('delay', delay_ms),
                  ('stim',[5],150,100), ('delay', delay_ms),
                  ('stim',[6],150,100), ('delay', delay_ms), 
                  ('stim',[7],150,100), ('delay', delay_ms),
                  ('stim',[8],150,100), ('delay', delay_ms),
                  ('stim',[9],150,100), ('delay', delay_ms), 
                  ('stim',[10],150,100), ('delay', delay_ms),
                  ('stim',[11],150,100), ('delay', delay_ms),
                  ('stim',[12],150,100), ('delay', delay_ms), 
                  ('stim',[13],150,100), ('delay', delay_ms),
                  ('stim',[14],150,100), ('delay', delay_ms),
                  ('stim',[15],150,100), ('delay', delay_ms),
                  ('stim',[16],150,100), ('delay', delay_ms),
                  ('stim',[17],150,100), ('delay', delay_ms), 
                  ('stim',[18],150,100), ('delay', delay_ms),
                  ('stim',[18],150,100), ('delay', delay_ms),
                  ('stim',[20],150,100), ('delay', delay_ms), 
                  ('stim',[21],150,100), ('delay', delay_ms),
                  ('stim',[22],150,100), ('delay', delay_ms),
                  ('stim',[23],150,100), ('delay', delay_ms), 
                  ('stim',[24],150,100), ('delay', delay_ms),
                  ('stim',[25],150,100), ('delay', delay_ms),
                  ('stim',[26],150,100), ('delay', delay_ms), 
                  ('stim',[27],150,100), ('delay', delay_ms),
                  ('stim',[28],150,100), ('delay', delay_ms),
                  ('stim',[29],150,100), ('delay', delay_ms),
                  ('stim',[30],150,100), ('delay', delay_ms),
                  ('stim',[31],150,100), ('delay', delay_ms), 
                  ('stim',[32],150,100), ('delay', delay_ms),
                  ('stim',[33],150,100), ('delay', delay_ms),
                  ('stim',[34],150,100), ('delay', delay_ms), 
                  ('stim',[35],150,100), ('delay', delay_ms),
                  ('stim',[36],150,100), ('delay', delay_ms),
                  ('stim',[37],150,100)]
                  '''


# Set up the variables
env = MaxwellEnv(**params)
done = False

fs = 20000
neurons = len(params['stim_electrodes'])

# Get the starting phase, and the time at which the next phase starts
phase, phase_dt = full_exp.pop(0) 
phase_start_time = env.time_elapsed()

# For bookkeeping
causal_count = 0

# Ray: bookkeeping
# repeat_count = 0

print('Starting experiment at {:.3f}'.format(env.time_elapsed()))
print('Beginning with phase',phase,'for',phase_dt,'seconds')

while not done:
    # ~~~~~~~~~~~ Phase changing Logic ~~~~~~~~~~~
    if phase_dt <= env.time_elapsed() - phase_start_time:
        # We are in the next phase
        if len(full_exp) == 0:
            phase, phase_dt = full_exp.pop(0)
        else:
            done = True
            print('Experiment complete')
            break
        phase_start_time = env.time_elapsed()
        print("Starting phase",phase,"at\t",'{:.3f}'.format(env.time_elapsed()), "for", phase_dt, "seconds")

        if phase == 'causal':
            # We reset the causal count
            causal_count = 0

    # ~~~~~~~~~~~ Silent Phase Logic ~~~~~~~~~~~
    if phase == 'silent':
        # We do nothing
        env.step()

    # ~~~~~~~~~~~ Causal Phase Logic ~~~~~~~~~~~
    elif phase == 'causal':
        # We stimulate the electrodes one-by-one at 1Hz 30 times
        if env.stim_dt >= 1/causal_Hz:
            action = ([causal_count//30],150,100) # Neuron, amplitude, phase duration
            print('Causal on {} at {:.3f}'.format(causal_count//30,env.time_elapsed()))
            env.step(action=action, tag='causal')
            causal_count += 1
        else:
            env.step()
    
    # ~~~~~~~~~~~ Tetanus Phase Logic ~~~~~~~~~~~
    elif phase == 'tetanus':
        
         '''Ray: # while repeat_count<=num_repeats:
                 if env.stim_dt >= 1/tetanus_Hz:
                    print('Tetanus at {:.3f}'.format(env.time_elapsed()))
                    env.step(action=tetanus_action2, tag='tetanus')
                    
                 else:
                     if repeat_count = num_repeats-1:
                         params['stim_electrodes'] = full_electrode_list[32*repeat_count:num_last_repeat]
                     else:
                         params['stim_electrodes'] = full_electrode_list[32*repeat_count:((32*repeat_count)+32)]
                         
                     env.update_params(**params)
                 
                 repeat_count+=1
             else:
             env.step()
        '''
            
        # We stimulate the electrodes one-by-one at 1Hz 30 times
        if env.stim_dt >= 1/tetanus_Hz:
            print('Tetanus at {:.3f}'.format(env.time_elapsed()))
            env.step(action=tetanus_action, tag='tetanus')
            
        else:
            env.step()
          

env._cleanup()