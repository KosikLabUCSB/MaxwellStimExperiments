import zmq
import struct
import time
import random
import numpy as np

from collections import namedtuple

context = zmq.Context()

# Define the ZMQ publishers
publisher_unfiltered = context.socket(zmq.PUB)
publisher_unfiltered.bind("tcp://*:7204")

publisher_filtered = context.socket(zmq.PUB)
publisher_filtered.bind("tcp://*:7205")

# SpikeEvent is a named tuple that represents a single spike event
SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')

_spike_struct = '8xLif'
_spike_struct_size = struct.calcsize(_spike_struct)

n_channels = 1024
fs = 20000
total_time_steps = 20000*10  # Total time steps to pre-generate
time_array = np.arange(total_time_steps)/fs  # Time array in seconds

# Pre-generate the frame data as a 2D array of sin waves for each channel
frame_data = np.sin(2*np.pi*1*time_array)  # 1 Hz sin wave
frame_data = np.tile(frame_data, (n_channels, 1))
frame_data = frame_data.astype(np.float32)

# Pre-generate random events for each time step
events_data = []
for t in range(total_time_steps):
    events_at_t = []
    n_events = random.randint(1, 5)  # Random number of events at this time step
    for i in range(n_events):
        event = SpikeEvent(frame=t, channel=random.randint(0, n_channels), amplitude=random.uniform(-50, -10))
        packed_event = struct.pack(_spike_struct, *(list(event)))
        events_at_t.append(packed_event)
    events_data.append(events_at_t)

print("Pre-generated data complete, beginning transmission...")
# Main loop
frame_number = 0
while True:
    # Get the data for this frame
    
    frame = frame_data[:, frame_number]
    events = events_data[frame_number]

    # Pack the frame number as a long long
    packed_frame_number = struct.pack('Q', frame_number)

    # Send the data
    publisher_unfiltered.send_multipart([packed_frame_number, frame.tobytes(), b''.join(events)])
    publisher_filtered.send_multipart([packed_frame_number, frame.tobytes(), b''.join(events)])

    # Increment the frame number, looping back to 0 when we reach the end
    frame_number = (frame_number + 1) % total_time_steps
