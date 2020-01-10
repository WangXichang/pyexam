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
    "mode_ratio_prox": 'lower_max',
    "mode_ratio_cumu": 'no',
    "mode_sort_order": 'd',
    "mode_section_point_first": 'real',
    "mode_section_point_start": 'step',
    "mode_section_point_last": 'real',
    "mode_section_degraded": 'to_max',
    "mode_section_lost": 'real',
    "mode_score_zero": 'real',
    "raw_score_range": (0, 100),
    }


# set other parameters
run_other_para = {
    "out_score_decimals": 0,
    "reload": False,
    "save_result": False,
    "path_name": "./",
    "display": True,
    "verify": False,
    "tiny_value": 10 ** -8,
    "logger": True
    }
