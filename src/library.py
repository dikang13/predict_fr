# Public libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns

# Flavell Lab packages 
import flv_utils as flv
from multianimalbleachcorrect import apply_bleach_correction