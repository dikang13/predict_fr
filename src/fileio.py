import pickle
import flv_utils as flv

def import_from_flv_utils(*,
    neuron_classes = ["M3"],
    confidence_threshold = 1,
    LR_inc = 'either',
    LR_ops = 'random'
):

    # SparseFood
    out1 = flv.by_multiclass(neuron_classes, 
                             tag_filter=['copper', 'tdc-1'], include=['neuropal', 'baseline'], exclude=['gfp'], 
                             confidence_threshold=confidence_threshold, 
                             LR_inc=LR_inc, LR_ops=LR_ops,
                             length_bounds = (1600,1600))

    out2 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1'], include=['neuropal', 'stim'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (800,800))


    # OffFood - JustFed
    out3 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'just_fed'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (1600,1600))

    out4 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'just_fed'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (800,800))

    # OffFood - Fasted
    out5 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'fasted'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (1600,1600))

    out6 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'fasted'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (800,800))

    # OffFood - 1hStarved
    out7 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'very_starved'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (1600,1600))

    out8 = flv.by_multiclass(neuron_classes, 
                            tag_filter=['copper', 'tdc-1', 'stim'], include=['neuropal', 'very_starved'], exclude=['gfp'], 
                            confidence_threshold=confidence_threshold, 
                            LR_inc=LR_inc, LR_ops=LR_ops,
                            length_bounds = (800,800))
    
    data = {}
    data['all']         = [out1, out2, out3, out4, out5, out6, out7, out8]
    data['sparse_food'] = [out1, out2]
    data['off_food']    = [out3, out4, out5, out6, out7, out8]
    data['just_fed']    = [out3, out4]
    data['fasted']      = [out5, out6]
    data['1h_starved']  = [out7, out8]

    return data


def save_dict_as_pickle(dictionary, filename):
    """Save dictionary as pickle file (binary, preserves Python objects)"""
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved as pickle: {filename}")
    
def load_dict_from_pickle(filename):
    """Load dictionary from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)