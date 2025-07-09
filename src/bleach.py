import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["JAX_DEBUG_NANS"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from multianimalbleachcorrect import apply_bleach_correction

def correct_bleach(
    ys,   # (n_traces, n_timepoints)    
    neuron_class:str,
    list_neuron_end_in_LR = ['AVL', 'RIR', 'IL1L', 'IL1R', 'OLL', 'IL2L', 'IL2R', 'ADL', 'ASEL', 'ASER', 'AQR', 'SAADR']
):
    
    if (neuron_class[-1] == 'L' or neuron_class[-1] == 'R') and neuron_class not in list_neuron_end_in_LR:
        neuron_name = neuron_class[:-1]
    else:
        neuron_name = neuron_class
        
    return apply_bleach_correction(neuron_name, ys)                  # (n_traces, n_timepoints)    