from bleach import *
from events import *
import numpy as np
import bisect
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

def bootstrap_mean_std(traces, n_bootstrap=100):
    """Perform bootstrap sampling and return mean and std of bootstrapped means"""
    traces = np.array(traces)
    if traces.shape[0] == 0:
        return None, None

    n_samples = len(traces)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = traces[bootstrap_indices]
        bootstrap_means.append(np.nanmean(bootstrap_sample, axis=0))

    bootstrap_means = np.array(bootstrap_means)
    mean_of_means = np.median(bootstrap_means, axis=0)
    std_of_means = np.std(bootstrap_means, axis=0)

    return mean_of_means, std_of_means


def original_mean_std(traces):
    """Calculate mean and standard error of original traces"""
    traces = np.array(traces)
    if traces.shape[0] == 0:
        return None, None

    mean_trace = np.nanmedian(traces, axis=0)
    std_trace = np.nanstd(traces, axis=0)

    return mean_trace, std_trace


def get_run_start_eta(
    list_outs:list, neuron_classes:list[str],
    rev_thresh = -0.005, 
    denoise_size = 7,
    t_before = 8, 
    t_after  = 16,
    thresh_short_FWD = 17,  # <= this threshold, fwd run is short
    thresh_long_FWD  = 24,  # >= this threshold, fwd run is long
    thresh_short_REV = 10,  # <= this threshold, reversal is short
    thresh_long_REV  = 13,  # >= this threshold, reversal is long
    n_beh_classes=3, 
    n_neuron_classes=1
):
    
    # Gather all forward runs for that group
    animals_seen = set()
    beh_data    = []
    neu_data    = []
    rev_data    = []
    rev_data_2  = []
    target_data = []
    animal_data = []
    
    for (out_idx, out) in enumerate(list_outs):
        # Grab sample size
        all_animal_id = out['datasets']
        new_animal_id = list(set(all_animal_id) - animals_seen)
        new_animal_idx = [i for i, animal in enumerate(all_animal_id) if animal in new_animal_id]
        animals_seen.update(set(new_animal_id))

        # Grab forward run starting times
        behaviors = out['behavior_traces']
        vel = behaviors['velocity']
        denoised_vel = np.ones(vel.shape)
        fwd_rev  = np.ones(vel.shape)
        rev_fwd  = np.ones(vel.shape)

        # Grab neural traces
        processed_ys = {}
        for neuron_class in neuron_classes:
            neuron_traces = out['neuron_traces'][neuron_class]

            # Custom Z-score
            trace_baseline = np.quantile(neuron_traces, 0.2, axis=1, keepdims=True)   # (n_traces, 1)
            global_std = np.std(neuron_traces - trace_baseline)                       # scalar
            ys = (neuron_traces - trace_baseline) / global_std                        # (n_traces, n_timepoints)

            # Multi-animal bleach correction
            ys_unbleached = correct_bleach(ys, neuron_class)                          # (n_traces, n_timepoints)

            # Savitzky-Golay - smooths while preserving peaks/valleys
            ys_denoised = savgol_filter(ys_unbleached, window_length=3, polyorder=1)  # (n_traces, n_timepoints)
            processed_ys[neuron_class] = ys_denoised / 3 - 0.5               # squeeze into the range (-1,1) for model fitting


        for i in new_animal_idx:    ## just a subset of rows!!!!
            denoised_vel[i] = median_filter(vel[i], size=denoise_size)
            fwd_rev[i] = np.where(denoised_vel[i] > rev_thresh, 1, 0)  # 1 is fwd
            rev_fwd[i] = np.where(denoised_vel[i] > rev_thresh, 0, 1)  # 1 is rev
            animal_id  = all_animal_id[i]
            
            # Identify forward runs from denoised velocity
            fwd_start, fwd_end, fwd_dur = find_events(fwd_rev[i])
            rev_start, rev_end, rev_dur = find_events(rev_fwd[i])

            for j, f_dur in enumerate(fwd_dur):   ### each fwd run
                f_start = fwd_start[j]
                if f_start < t_before:  # check if there are enough time points before the event
                    continue

                # Classify forward runs into 3 groups
                if f_dur <= thresh_short_FWD:
                    fwd_label = 0  # short runs
                elif f_dur > thresh_short_FWD and f_dur < thresh_long_FWD:
                    fwd_label = 1  # medium runs
                else:  # f_dur > thresh_long
                    fwd_label = 2  # long runs
                    
                # Classify preceding reversals into 3 groups
                last_rev_idx = bisect.bisect_left(rev_end, f_start) -1 # find the index of REV event that preceded this FWD run
                if last_rev_idx < 0:  # there is no REV event beforehand
                    rev_label = -1 # not useful for categorization of FWD runs by preceding reversal
                else:
                    r_dur = rev_dur[last_rev_idx]   # how long the last REV event was
                    if r_dur <= thresh_short_REV:
                        rev_label = 0 # short reversal
                    elif r_dur > thresh_short_REV and r_dur < thresh_long_REV:
                        rev_label = 1 # medium reversal
                    else:
                        rev_label = 2 # long reversal

                # get input snippets
                start_idx = f_start - t_before
                end_idx   = f_start + t_after

                # Handle case where duration is shorter than t_after
                actual_end_idx = min(end_idx, f_start + f_dur)
                t_range = slice(start_idx, actual_end_idx)

                n_t = t_before + t_after 
                beh_snippets = np.full((n_beh_classes, n_t), np.nan)
                neu_snippets = np.full((n_neuron_classes, n_t), np.nan)

                # Fill available data
                available_length = actual_end_idx - start_idx

                for k, beh in enumerate(list(behaviors.keys())):
                    beh_snippets[k, :available_length] = behaviors[beh][i, t_range]

                for k, neu in enumerate(neuron_classes):
                    neu_snippets[k, :available_length] = processed_ys[neuron_class][i, t_range]

                # append
                beh_data.append(beh_snippets)
                neu_data.append(neu_snippets)
                rev_data.append(rev_label)
                rev_data_2.append(r_dur)
                target_data.append(fwd_label)
                animal_data.append(animal_id)
    
    beh_data    = np.array(beh_data).transpose(1,0,2)
    neu_data    = np.array(neu_data).transpose(1,0,2)
    rev_data    = np.array(rev_data)
    rev_data_2  = np.array(rev_data_2)
    target_data = np.array(target_data)
    animal_data = np.array(animal_data)
    
    return beh_data, neu_data, rev_data, rev_data_2, target_data, animal_data