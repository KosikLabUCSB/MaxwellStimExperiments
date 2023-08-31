# maxwell_env.py
import sys
import zmq
import struct
import array
import time
import numpy as np
import os

from pathlib import Path
from multiprocessing import Process, Queue

dummy = False
try:
    import maxlab
    import maxlab.system
    import maxlab.chip
    import maxlab.util
    import maxlab.saving
except ImportError:
    print("No maxlab found, instead using dummy maxlab module!")
    dummy = True
    import core.dummy_maxlab as maxlab

from collections import namedtuple
from csv import writer

from .base_env import BaseEnv

# SpikeEvent is a named tuple that represents a single spike event
SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')

_spike_struct = '8xLif'
_spike_struct_size = struct.calcsize(_spike_struct)
fs_ms = 20 # sampling rate in kHz


class MaxwellEnv(BaseEnv):
    """
    The MaxwellEnv class extends from the BaseEnv class and implements a specific environment 
    for running Maxwell's equations simulations.


    Attributes
    ----------
    config : str
        Stores the config filepath in order to easily reload the array.
    name : str
        Stores the name of the environment instance.
    max_time_sec : int
        Stores the maximum experiment time.
    save_file : str
        The file where the data will be saved.
    stim_electrodes : list
        Stores the list of electrodes for stimulation.
    verbose : int
        Controls the verbosity of the environment's operations.
    array : None
        Initialized as None, to be updated in sub-classes as needed.
    subscriber : None
        Initialized as None, to be updated in sub-classes as needed.
    save_dir : str
        Stores the directory where the simulation data will be saved.
    is_stimulation : bool
        A flag that indicates whether a stimulation is going to occur.
    stim_log_file : str or None
        The file where the log of the stimulation is saved. If no stimulation is going to occur, this is None.
    stim_units : None
        Initialized as None, to be updated in sub-classes as needed.
    stim_electrodes_dict : None
        Initialized as None, to be updated in sub-classes as needed.
    start_time : float
        The time when the environment is initialized.
    cur_time : float
        The current time, updated at each step.
    last_stim_time : float
        The time when the last stimulation occurred.
    smoke_test : bool
        A flag that indicates whether the environment is being used for a smoke test.
        
    """

    def __init__(self, config, name="", stim_electrodes=[], max_time_sec=60,
                save_dir="data", multiprocess=False, render=False,
                filt=False, verbose = 1, smoke_test=False):
        """
        Parameters
        ----------
        config : str
            A path to the maxewll config file. This is usually made by the Maxwell GUI, 
            and contains the information about the array.
        name : str
            The name of the environment instance. This is used for saving data.
        stim_electrodes : list
            A list of electrodes for stimulation. If no electrodes are specified, no stimulation will occur.
        max_time_sec : int
            The maximum experiment time in seconds.
        save_dir : str
            The directory where the stimulation data will be saved.
        filt : bool
            A flag that indicates whether a filter should be applied to the data. The filter is onboard the chip,
            and is applied to the data before it is sent to the computer. It adds ~100ms of latency.
        verbose : int
            An integer that controls the verbosity of the environment's operations. 0 is silent, 1 is verbose.
        smoke_test : bool
            A flag that indicates whether the environment is being used for a smoke test. If True, the environment
            will not save any data, will use dummy logic, and no hardware will be used.
        """


        self.config = config
        self.name = name
        self.max_time_sec = max_time_sec
        self.multiprocess = multiprocess

        self.stim_electrodes = stim_electrodes
        self.active_units = []    
        
        self.verbose = verbose
        self.array = None
        self.subscriber = None

        # Setup saving
        self.save_dir = str(Path(save_dir).resolve())
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_file = os.path.join(self.save_dir, f'{name}')

        self._init_save()

        # Check whether stimulation will occur
        if len(stim_electrodes) == 0:
            self.is_stimulation = False
            self.stim_log_file = None
        else:
            self.is_stimulation = True
            self.stim_log_file = os.path.join(save_dir, name + '_log.csv')
            self._init_log_stimulation()

        
        # Ray: here stim electrodes are selected
        
        self.subscriber, self.stim_units, self.stim_electrodes_dict = init_maxone(
                config, stim_electrodes, filt=filt,
                verbose=1, gain=512, cutoff='1Hz',
                spike_thresh=5
        )

        if self.multiprocess:
            subscriber_args = (filt, verbose)
            self.data_queue = Queue()
            self.event_queue = Queue()
            self.worker = Process(target=socket_worker, args=(self.data_queue, self.event_queue, subscriber_args))
            self.worker.start()

            if render:
                self.plot_worker = Process(target=plot_worker, args=(self.data_queue,))
                self.plot_worker.start()


        if not dummy:
            time.sleep(5) # Wait for the system to settle

        # Time management
        self.start_time = self.cur_time = time.perf_counter()
        self.last_stim_time = 0

        print("===================== Beginning experiment =====================")


    def reset(self):
        """
        Reset the environment
        """

        # Close previous save files
        self.stim_log_file.close()
        
        # Properly increment filename
        if self.name[-1].isdigit(): 
            self.name = self.name[:-1] + str(int(self.name[-1])+1)
        else:
            self.name += '_0'

        if self.is_stimulation:
            self.stim_log_file = f'{self.save_dir}{self.name}_log.csv'
            self._init_log_stimulation()

        # init_maxwell() #TODO: Reset the array
        self.start_time = self.cur_time = time.perf_counter()
        self.last_stim_time = 0

     # Ray: Add update_params function
        ''' def update_electrodes(self, **params):    
                for param, value in params.items():
                    if hasattr(self, param) and value is not self.param:
                        setattr(self, param, value)
        #print out changes if verbose
        
             '''
    
    def step(self, action=None, tag=None):
        '''
        Recieve events published since last time step() was called.
        This includes spike events and raw datastream.

        Parameters
        ----------
        action : list 
            A list of stimulation commands. Each command is a tuple of the form (electrode_index, amplitude_mV, phase_length_us).
        '''
        self.cur_time = time.perf_counter()

        # Receive data
        if self.multiprocess:
            # frame_number, frame_data, events_data = 
            obs = self.event_queue.get()

        else:
            frame_number, frame_data, events_data = receive_packet(self.subscriber)#TODO: Get all frames, populate buffer
            # frame = self._parse_frame(frame_data) # Raw datastream
            obs = parse_events_list(events_data) # Spike events


        if action is not None:
            if type(action[0][0]) == str:
                self._create_stim_pulse_sequence(action)
            else: 
                self._create_stim_pulse(action)

            self.seq.send()
            self._log_stimulation(action, tag=tag)
            self.last_stim_time = self.cur_time

            if self.verbose >=2:
                print(f'Stimulating at t={self.cur_time} with command:', self.seq.token)


        done = self._check_if_done()

        return obs, done
    
    @property
    def dt(self):
        '''Returns time since the last step.'''
        return time.perf_counter() - self.cur_time
    
    @property
    def stim_dt(self):
        '''Returns time since last stimulation.'''
        return time.perf_counter() - self.last_stim_time

    def time_elapsed(self):
        '''Returns time since initialization of the environment.'''
        return time.perf_counter() - self.start_time
    

    def _check_if_done(self):
        if self.time_elapsed() > self.max_time_sec:
            # Debugging
            if self.verbose >=1:
                print(f'Max time {self.max_time_sec} reached at {self.time_elapsed()}')
            self._cleanup()

            return True
        return False
    

    def _create_stim_pulse(self, stim_command):
        '''
        Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief
        period. This should be combined with code that connects electrodes to the DAC in order
        to actually generate stimulus behavior.

        Parameters
        ----------
        stim_command : tuple
            A tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
                stim_electrodes : list
                    A list of electrode numbers to stimulate.
                amplitude_mV : float
                    The amplitude of the square wave, in mV.
                phase_length_us : float
                    The length of each phase of the square wave, in us.
        '''
        self.seq = maxlab.Sequence()
        self.active_units = []
        
        neurons, amplitude_mV, phase_length_us = stim_command # phase length in us
        
        # Append each neuron that needs to be stimmed in order
        for n in neurons:
            # print(f'Adding neuron {n} to stim sequence')
            unit = self.stim_units[n]
            self.active_units.append(unit)
            self.seq.append(unit.power_up(True))
        
        # Create pulse
        self._insert_square_wave(amplitude_mV, phase_length_us)

        # Power down all units
        for unit in self.active_units:
            self.seq.append(unit.power_up(False))

        return self.seq

    def _create_stim_pulse_sequence(self, stim_commands):
        '''
        Create a pulse sequence that just sets the DAC amplitude in a pulse shape for a brief
        period. This should be combined with code that connects electrodes to the DAC in order
        to actually generate stimulus behavior.

        Parameters
        ----------
        stim_commands : list of tuples
            A tuple of the form (command, stim_electrodes, amplitude_mV, phase_length_us)
                stim_electrodes : list
                    A list of electrode numbers to stimulate.
                amplitude_mV : float
                    The amplitude of the square wave, in mV.
                phase_length_us : float
                    The length of each phase of the square wave, in us.

        ------------------------------------------------
        For 'stim' command:
        ('stim', [neuron inds], mv, us per phase)

        For 'delay'
        ('delay', frames_delay)
        
        For 'next'
        ('next', None)
        This command acts as a placeholder to move to the next timepoint in the time_arr or the next
        period triggered by the freq_Hz
        -------------------------------------------------
        '''
        self.seq = maxlab.Sequence()
        self.active_units = []
        stim_commands = stim_commands.copy()

        # Build the sequence
        command = None
        while len(stim_commands) > 0:
            command, *params = stim_commands.pop(0)
            
            # ----------------- stim --------------------
            if command == 'stim':
                neurons, amplitude_mV, phase_length_us = params # phase length in us
                
                # Append each neuron that needs to be stimmed in order
                for n in neurons:
                    unit = self.stim_units[n]
                    self.active_units.append(unit)
                    self.seq.append(unit.power_up(True))
                
                # Create pulse
                self._insert_square_wave(amplitude_mV, phase_length_us)

                # Power down all units
                for unit in self.active_units:
                    self.seq.append(unit.power_up(False))
            
            # ----------------- delay --------------------
            if command == 'delay':
                self.seq.append( maxlab.system.DelaySamples(params[0]*fs_ms))
                
            # ----------------- next --------------------
            if command == 'next':
                break 

        return self.seq
        
        neurons, amplitude_mV, phase_length_us = stim_command # phase length in us
        
        # Append each neuron that needs to be stimmed in order
        for n in neurons:
            # print(f'Adding neuron {n} to stim sequence')
            unit = self.stim_units[n]
            self.active_units.append(unit)
            self.seq.append(unit.power_up(True))
        
        # Create pulse
        self._insert_square_wave(amplitude_mV, phase_length_us)

        # Power down all units
        for unit in self.active_units:
            self.seq.append(unit.power_up(False))

        return self.seq



    def _insert_square_wave(self, amplitude_mV = 150, phase_length_us = 100):
        '''
        Adds a square wave to the sequence with a set amplitude and duty cycle.

        Parameters
        ----------
        amplitude_mV : float
            The amplitude of the square wave, in mV.

        duty_time_us : float
            The duty cycle of the square wave, in us.
            We can only set the duty cycle in increments of 50us, so this will be rounded.

        '''
        amplitude_lsbs = round(amplitude_mV / 2.9) # scaling factor given by Maxwell
        duty_time_samp = round(phase_length_us * .02)

        self.seq.append( maxlab.chip.DAC(0, 512 - amplitude_lsbs) )
        self.seq.append( maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( maxlab.chip.DAC(0, 512 + amplitude_lsbs) )
        self.seq.append( maxlab.system.DelaySamples(duty_time_samp) )
        self.seq.append( maxlab.chip.DAC(0, 512) )

    
    #==========================================================================
    #=========================  Saving Functions  =============================
    #==========================================================================
    def _init_save(self):
        '''
        Initialize the save file for the environment.
        Saved in self.save_dir with name self.name.
        '''
        self.saver = maxlab.saving.Saving()
        self.saver.open_directory(self.save_dir)
        self.saver.set_legacy_format(False)
        self.saver.group_delete_all()
        self.saver.group_define(0, "routed")
        self.saver.start_file(self.name)
        self.saver.start_recording([0])


    def _init_log_stimulation(self):
        if self.stim_log_file is not None:
            self.stim_log_file = open(self.stim_log_file, 'a+', newline='')
            self.stim_log_writer = writer(self.stim_log_file)
            # write first row: stim time, amplitude
            self.stim_log_writer.writerow(['time', 'amplitude', 'duty_time_ms', 'stim_electrodes', 'tag'])

    def _log_stimulation(self, stim_command, tag=None):
        '''
        Log the stimulation command to a csv file.
        Stim command is a tuple of the form (stim_electrodes, amplitude_mV, phase_length_us)
        '''
        if self.stim_log_file is not None:
            if tag is None:
                tag = ''
            if type(stim_command[0][0]) == str:
                # We just write the first one since it becomes too complicated to write all of them
                elecs = []
                for cmd in stim_command:
                    if cmd[0] == 'stim':
                        elecs.append([self.stim_electrodes[i] for i in cmd[1]])
                self.stim_log_writer.writerow([self.time_elapsed(),
                                        stim_command[0][1], stim_command[0][2], elecs, tag])
            else:
                elecs = [self.stim_electrodes[i] for i in stim_command[0]]
                self.stim_log_writer.writerow([self.time_elapsed(),
                                        stim_command[1], stim_command[2], elecs, tag])

    def _cleanup(self):
        '''Shuts down the environment and saves the data.'''
        self.saver.stop_recording()
        self.saver.stop_file()
        self.saver.group_delete_all()
        if self.stim_log_file is not None:
            self.stim_log_file.close()
        





#==========================================================================
#=========================  MAXWELL FUNCTIONS  ============================
#==========================================================================

def init_maxone(config, stim_electrodes,filt=True, verbose=1, gain=512, cutoff='1Hz',
                spike_thresh=5):
    """
    Initialize MaxOne, set electrodes, and setup subscribers

    Parameters
    ----------
    config : str
        Path to the config file for the electrodes

    stim_electrodes : list  
        List of electrode numbers to stimulate

    filt : bool
        Whether to use the high-pass filter

    verbose : int   
        0: No print statements
        1: Print initialization statements
        2: Print all statements

    gain : int, {512, 1024, 2048}   
        Gain of the amplifier

    cutoff : str, {'1Hz', '300Hz'}  
        Cutoff frequency of the high-pass filter

    spike_thresh : int  
        Threshold for spike detection, in units of standard deviations
    """

    init_maxone_settings(gain=gain, cutoff=cutoff, spike_thresh=spike_thresh, verbose=verbose)
    subscriber = setup_subscribers(filt=filt, verbose=verbose)
    stim_units, stim_electrodes_dict = select_electrodes(config,
                                                        stim_electrodes, verbose=verbose)
    ignore_first_packet(subscriber)
    return subscriber, stim_units, stim_electrodes_dict
        
        



def init_maxone_settings(gain=512,amp_gain=512,cutoff='1Hz', spike_thresh=5, verbose=1):
    """
    Initialize MaxOne and set gain and high-pass filter

    Parameters
    ----------
    gain : int, {512, 1024, 2048}

    amp_gain : int

    cutoff : str, {'1Hz', '300Hz'}
    """
    if verbose >= 1:
        print("Initializing MaxOne")

    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(amp_gain))
    maxlab.send_raw(f"stream_set_event_threshold {spike_thresh}")
    # maxlab.util.set_gain(gain)
    # maxlab.util.hpf(cutoff)
    if verbose >=1:
        print('MaxOne initialized')

        



def setup_subscribers(filt, verbose=1):
    """
    Setup subscribers for events from MaxOne, this 
    allows us to read the data from the chip.
    """
    if verbose >= 1:
        print("Setting up subscribers")
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 0)
    subscriber.setsockopt(zmq.RCVBUF, 10*20000*1030)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    subscriber.setsockopt(zmq.RCVTIMEO, 100)
    if filt:
        subscriber.connect('tcp://localhost:7205')
    else:
        subscriber.connect('tcp://localhost:7204')
    return subscriber


def select_electrodes(config, stim_electrodes, verbose=1):
    # Electrode selection logic
    array = maxlab.chip.Array('stimulation')
    array.reset() #delete previous array

    array.load_config(config)

    if verbose >= 1:
        print(f'Recording electrodes initialized from: {config}')

    array.select_stimulation_electrodes(stim_electrodes)

    if len(stim_electrodes) > 32:
        raise Exception('Too many stimulation electrodes.')

    stim_units = []
    stim_electrodes_dict = {}
    for e in stim_electrodes:
        array.connect_electrode_to_stimulation(e)
        unit_id = array.query_stimulation_at_electrode(e)
        if not unit_id:
            print(f"Error: electrode: {e} cannot be stimulated")

        unit = maxlab.chip.StimulationUnit(unit_id)

        stim_units.append(unit)
        stim_electrodes_dict[unit] = e

        if verbose >= 2:
            print(f'Connected Electrode # {e}')
    array.download()

    if verbose >= 1:
        print(f'Electrodes selected for stimulation: {stim_electrodes}')

    power_cycle_stim_electrodes(stim_units)
    maxlab.util.offset()
    return stim_units, stim_electrodes_dict




def power_cycle_stim_electrodes(stim_units):
    ''' "Power up and down again all the stimulation units.
    It appears this is needed to equilibrate the electrodes"
    - from maxwell code'''

    # Ray: Can this be done in one for loop: power up, then power down, so as not to excede the max # of electrodes?
    
    seq = maxlab.Sequence()
    for unit in stim_units:
        seq.append(
                unit.power_up(True).connect(True)
                    .set_voltage_mode().dac_source(0))
    for unit in stim_units:
        seq.append(unit.power_up(False).connect(False))
    seq.send()
    print('Power cycled')
    del seq
    seq = maxlab.Sequence()
    
    # Ray: Then remove this step and only power-on and connect 1 electrode at a time per step.
    
    for unit in stim_units:
        seq.append(
                unit.power_up(True).connect(True))
    seq.send()


def parse_events_list(events_data):
    '''
    Parse the raw binary events data into a list of SpikeEvent objects.
    '''
    events = []

    if events_data is not None:
        # The spike structure is 8 bytes of padding, a long frame
        # number, an integer channel (the amplifier, not the
        # electrode), and a float amplitude.

        if len(events_data) % _spike_struct_size != 0:
            print(f'Events has {len(events_data)} bytes,',
                f'not divisible by {_spike_struct_size}', file=sys.stderr)

        # Iterate over consecutive slices of the raw events
        # data and unpack each one into a new struct.
        for i in range(0, len(events_data), _spike_struct_size):
            ev = SpikeEvent(*struct.unpack(_spike_struct,
                events_data[i:i+_spike_struct_size]))
            events.append(ev)

    return events
    
    
def parse_frame(frame_data):
    '''
    Parse the binary frame data into an array of floating-point voltages.
    '''
    return None if frame_data is None else array.array('f',frame_data)


def receive_packet(subscriber):
    '''
    Use the subscriber to capture the frame and event data from the server.
    Returns an integer frame_number as well as data buffers for the voltage
    data frame and the spike events. Also sets the current time.
    '''
    frame_number = frame_data = events_data = None

    # Sometimes the publisher will be interrupted, so fail cleanly by
    # terminating this run of the environment, returning done = True.
    try:
        # The first component of each message is the frame number as
        # a long long, so unpack that.
        frame_number = struct.unpack('Q', subscriber.recv())[0]

        # We're still ignoring the frame data, but we have to get it
        # out from the data stream in order to skip it.
        if subscriber.getsockopt(zmq.RCVMORE):
            frame_data = subscriber.recv()

        # This is the one that stores all the spikes.
        if subscriber.getsockopt(zmq.RCVMORE):
            events_data = subscriber.recv()

    except Exception as e:
        print(e)

    return frame_number, frame_data, events_data


def socket_worker(data_queue, event_queue, subscriber_args):
    """Worker function that reads from the ZeroMQ socket."""
    subscriber = setup_subscribers(*subscriber_args)
    while True:
        frame_number, frame_data, events_data = receive_packet(subscriber)
        
        if events_data is not None:
            event_queue.put(parse_events_list(events_data))
        if frame_data is not None:
            data_queue.put(parse_frame(frame_data))


def plot_worker(queue):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib
    matplotlib.use('TkAgg')
    # Set up plot
    fig, ax = plt.subplots()
    cur_data = np.zeros((2,20000))# 1s
    line, = ax.plot(cur_data[0,:])  # Adjust as needed
    ax.set_ylim([-1,1])

    # Animation update function
    def update(i):
        nonlocal cur_data
        for i in range(2000): #100ms
            if not queue.empty():
                data = np.array(queue.get())
                cur_data = np.roll(cur_data, -1, axis=1)
                cur_data[:2,-1] = data[:2]
            # filtered_data = butter_bandpass_filter(data, lowcut=1, highcut=50, fs=20000)  # Filter parameters to be adjusted
        line.set_ydata(cur_data[0,:])
        # ax.relim()  # Recalculate limits
        # ax.autoscale_view(True, True, True)  # Autoscale the plot
        ax.set_title(f'Frame {i}')
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, interval=20, blit=True)

    # Show plot
    plt.show()



def ignore_first_packet(subscriber, verbose=1):
    '''
    This first loop ignores any partial packets to make sure the real
    loop gets aligned to an actual frame. First it spins for as long
    as recv() fails, then it waits for the RCVMORE flag to be False to
    check that the last partial frame is over.
    '''
    more = True
    t = time.perf_counter()
    while more:
        try:
            _ = subscriber.recv()
        except zmq.ZMQError:
            if time.perf_counter() - t >= 3:
                raise TimeoutError("Make sure the Maxwell Server is on.")
            continue
        more = subscriber.getsockopt(zmq.RCVMORE)

    if verbose >=1:
        print('Successfully ignored first packet')