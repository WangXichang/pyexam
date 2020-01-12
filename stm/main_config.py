# coding: utf-8


# user work
# scripts for prepare data
# import pandas as pd
# df = pd.read_csv(filename)


# set model and data
model_name = 'shandong'     # must in main_in.Models or main.ext.Models
df = None                   # must be a pandas.DataFrame
cols = None                 # must be list with score field as elements, ['wl', 'hx', 'sw']


# set strategy items
run_strategy = {
    "mode_ratio_prox": 'lower_max',         # ('upper_min', 'lower_max', 'near_max', 'near_min')
    "mode_ratio_cumu": 'no',                # ('yes', 'no')
    "mode_sort_order": 'd',                 # ('d', 'a')
    "mode_section_point_first": 'real',     # ('real', 'defined')
    "mode_section_point_start": 'step',     # ('step', 'share')
    "mode_section_point_last": 'real',      # ('real', 'defined')
    "mode_section_degraded": 'to_max',      # ('to_max', 'to_min', 'to_mean')
    "mode_section_lost": 'real',            # ('real', 'zip')
    "mode_score_zero": 'real',              # ('real', 'after', 'alone')
    }


# set other parameters
run_parameters = {
    "display": True,
    "logging": True,
    "verify": False,
    "raw_score_range": (0, 100),
    "out_score_decimals": 0,
    "tiny_value": 10 ** -8,
    }
