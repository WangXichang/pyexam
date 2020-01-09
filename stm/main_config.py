# coding: utf-8


run_model_name = 'shandong'     # must in main_in.Models or main.ext.Models
run_df = None                   # must be a pandas.DataFrame
run_cols = None                 # must be list with score field as elements, ['wl', 'hx', 'sw']


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


run_other_para = {
    "out_score_decimals": 0,
    "reload": False,
    "save_result": False,
    "path_name": None,
    "display": False,
    "verify": False,
    "tiny_value": 10 ** -8
}
