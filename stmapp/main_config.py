# coding: utf-8


# user scripts for prepare data
# import pandas as pd
# df = pd.read_csv(filename)


# set model
model_name = 'shandong'     # must in main_in.Models or main.ext.Models

# set data
df = None                   # pandas.DataFrame
cols = None                 # list with score fields as elements, ['wl', 'hx', 'sw']

# set strategy items
run_strategy = {
    "mode_ratio_prox": 'upper_min',         # ('upper_min', 'lower_max', 'near_max', 'near_min')
    "mode_ratio_cumu": 'no',                # ('yes', 'no')
    "mode_sort_order": 'd',                 # ('d', 'a')
    "mode_section_point_first": 'real',     # ('real', 'defined')
    "mode_section_point_start": 'step',     # ('step', 'share')
    "mode_section_point_last": 'real',      # ('real', 'defined')
    "mode_section_degraded": 'to_max',      # ('to_max', 'to_min', 'to_mean')
    "mode_section_lost": 'real',            # ('real', 'zip')
    }

# set parameters
run_parameters = {
    "logname": '',
    "logdisp": 1,
    "logfile": 0,
    "verify": 1,
    "raw_score_range": (0, 100),
    "out_score_decimals": 0,
    "tiny_value": 10 ** -12,
    }
