import numpy as np

#Network Configuration
#each board communicate through PCIe interconnection
PCIe_bandwidth = 64 #GB/s
PCIe_latency = 5000 #nseconds

#each chip communicate through PCB trace interconnection
PCB_trace_bandwidth = 50 #GB/s
PCB_trace_latency = 0 # ignorable

# assume full-duplex communication

# Function to estimate communication overhead
def estimate_communication_overhead(bandwidth, latency, data_volume, pattern, p, overlap_fraction = 0.0):

    # Base communication time(ns) considering bandwidth and data volume(Bytes)

    # Adjust time by communication pattern
    if pattern == 'reduce-scatter':
        comm_time = np.log2(p) * latency + data_volume * (p - 1) / (p *  bandwidth)
    elif pattern == 'all-gather':
        comm_time = np.log2(p) * latency + data_volume * (p - 1) / (p * bandwidth)
    elif pattern == 'point-to-point':
        comm_time =  latency + data_volume / bandwidth # Point-to-point does not scale with number of devices
    elif pattern == 'all-reduce':
        comm_time = 2 * (np.log2(p) * latency + data_volume * (p - 1) / (p * bandwidth))
    else:
      print("non-existing communication pattern")
      exit(-1)
    # Additional scaling factor to model increased complexity with larger number of devices

    comm_time *= (1 - overlap_fraction)

    return comm_time

def estimate_communication_overhead_multi_bandwidth(bandwidth_low, latency_low, p_low, bandwidth_high, latency_high, p_high, data_volume, pattern, overlap_fraction = 0.0):

    # Base communication time(ns) considering bandwidth and data volume(Bytes)

    comm_time = 0
    # Adjust time by communication pattern
    if pattern == 'reduce-scatter':
        for i in range(int(np.log2(p_low))):
          comm_time += latency_low + data_volume /(2 ** (i + 1)) / bandwidth_low
        for i in range(int(np.log2(p_high))):
          comm_time += latency_high + data_volume /(2 ** (i + 1)) / bandwidth_high
    elif pattern == 'all-gather':
        for i in range(int(np.log2(p_low))):
          comm_time += latency_low + data_volume /(2 ** (i + 1)) / bandwidth_low
        for i in range(int(np.log2(p_high))):
          comm_time += latency_high + data_volume /(2 ** (i + 1)) / bandwidth_high
    elif pattern == 'all-reduce':
        for i in range(int(np.log2(p_low))):
          comm_time += latency_low + data_volume /(2 ** (i + 1)) / bandwidth_low
        for i in range(int(np.log2(p_high))):
          comm_time += latency_high + data_volume /(2 ** (i + 1)) / bandwidth_high

        comm_time *= 2 #it execute both reduce-scatther and all-gather
    else:
      print("non-existing communication pattern")
      exit(-1)
    # Additional scaling factor to model increased complexity with larger number of devices

    comm_time *= (1 - overlap_fraction)

    return comm_time

def cost_communication(num_layer, d_model, output_size, batch_size, chips_per_board, TP, PP):

  communication_latency = 0
  data_volume = int(batch_size / PP) * d_model * 2 #matrix col x row x 2bytes(BF16)

  # GENERATION
  for current_seq in range(output_size):
    #there are two all-reduce operation in decorder block according to Megatron-LM
    for i in range(2):
      if TP > chips_per_board:
        communication_latency += estimate_communication_overhead_multi_bandwidth(PCB_trace_bandwidth, PCB_trace_latency, chips_per_board, PCIe_bandwidth, PCIe_latency, int(TP / chips_per_board), data_volume, "all-reduce")
      else:
        communication_latency += estimate_communication_overhead(PCB_trace_bandwidth, PCB_trace_latency, data_volume, "all-reduce", TP)

  communication_latency *= num_layer

  return communication_latency

