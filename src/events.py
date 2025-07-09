import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

def find_events(binary_arr, verbose=False):
    # Find where transitions occur (0->1 and 1->0)
    diff_arr = np.diff(np.concatenate(([0], binary_arr, [0])))
    
    # Find start and end indices of events
    event_starts = np.where(diff_arr == 1)[0]  # 0->1 transitions
    event_ends = np.where(diff_arr == -1)[0]   # 1->0 transitions
    
    # Calculate durations
    event_durations = event_ends - event_starts
    
    if verbose:
        print("Event starts:", event_starts)      
        print("Event ends:", event_ends)          
        print("Event durations:", event_durations)    
    return event_starts, event_ends, event_durations


def get_fwd_rev_durations(
    list_outs:list, 
    rev_thresh = -0.005, 
    denoise_size = 7
):
    
    # Gather all forward runs and reversals
    data_dict = {}
    fwd_durations = []
    rev_durations = []
    first_event  = []
    animals_seen = []
    
    for (j, out) in enumerate(list_outs):
        # Grab sample size
        new_animal_id  = list(set(out['datasets']) - set(animals_seen))
        new_animal_idx = [i for i, animal in enumerate(out['datasets']) if animal in new_animal_id]
        # data_dict[group_name][j] = new_animal_idx   # for debugging only
        
        # Grab behavior traces
        behaviors = out['behavior_traces']

        # Compute binary fwd_rev from denoised velocity
        vel = behaviors['velocity']
        denoised_vel = np.ones(vel.shape)
        fwd_rev  = np.ones(vel.shape)
        rev_fwd  = np.ones(vel.shape)

        for i in new_animal_idx:    
            denoised_vel[i] = median_filter(vel[i], size=denoise_size)
            fwd_rev[i] = np.where(denoised_vel[i] > rev_thresh, 1, 0)  # 1 is forward
            rev_fwd[i] = np.where(denoised_vel[i] > rev_thresh, 0, 1)  # 1 is rev

            # Identify forward runs from denoised velocity
            fwd_start, fwd_end, fwd_dur = find_events(fwd_rev[i])
            fwd_durations.append(fwd_dur)
            
            # Identify reverals from denoised velocity
            rev_start, rev_end, rev_dur = find_events(rev_fwd[i])
            rev_durations.append(rev_dur)
            
            # Keep track of whether forward run happens first
            if fwd_start[0] < rev_start[0]:
                first_event.append('fwd')
            else:
                first_event.append('rev')
        
        # keep track of all animals seen in the get_run_durations attempt
        animals_seen.extend(new_animal_id)

    data_dict['fwd_durations'] = fwd_durations
    data_dict['rev_durations'] = rev_durations
    data_dict['first_event']   = first_event
    data_dict['animal_uid']    = animals_seen

    return data_dict
