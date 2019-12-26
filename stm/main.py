# coding: utf-8

import pandas as pd

from stm import modelconfig as mcfg
from stm import modelbase as mbas
from stm import modellib as mlib


# interface to use model for some typical application
def run(
        name='shandong',
        df=None,
        cols=(),
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='descending',
        raw_score_range=(0, 100),
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        out_decimal_digits=0
        ):
    """
    :param name: str, model name,
                 values: 'shanghai', 'zhejiang', 'beijing', 'tianjin',  # plt model for grade score
                         'shandong', 'guangdong', 'SS7',                # plt model for linear mapping score
                         'hn900', 'hn300',                              # ppt model to transform score from (0,100)-->(60, 300)
                         'hn300plt1', 'hn300plt2', 'hn300plt3'          # plt model to transform score from (0,100)-->(60, 300)
                         'zscore', 'tscore', 'tlinear'                  # ppt model for z, t, t-linear transform score
                 default = 'shandong'
    :param df: DataFrame,
       values: raw score data, instance of pandas.DataFrame, including score field, which type must be int or float
      default= None
    :param cols: list,
         values: [column name, that is score field name of df]
                 default = None
    :param mode_ratio_prox: str,
                  strategy: how to find endpoint by ratio
                values set: 'lower_max', 'upper_min', 'near_max', 'near_min'
                   default= 'upper_min'
    :param mode_ratio_cumu: str,
                  strategy: cumulate ratio or not
                values set: 'yes', 'no'
                   default= 'no'
    :param mode_sort_order: string,
                  strategy: which score order to search ratio
                    values: 'descending', 'ascending'
                   default= 'descending'
    :param mode_section_degraded: str,
                        strategy: how to map raw score when segment is one-point, [a, a]
                          values: 'map_to_max', map to max value of out score section
                                  'map_to_min', map to min value of out score section
                                  'map_to_mean', map to mean value of out score section
                         default= 'map_to_max'
    :param mode_section_point_first: str,
                           strategy: how to set first point of first section
                             values: 'real', use real raw score max or min value
                                     'defined', use test paper full score or least score
                            default= 'real_max_min'
    :param mode_section_point_start: str,
                           strategy: how to set first point of first section
                             values: 'real', use real raw score max or min value
                                     'defined', use test paper full score or least score
                            default= 'real_max_min'
    :param mode_section_point_last: str,
                           strategy: how to set first point of first section
                             values: 'real', use real raw score max or min value
                                     'defined', use test paper full score or least score
                            default= 'real_max_min'
    :param raw_score_range: tuple,
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= (0, 100)
    :param out_decimal_digits: int,
                        usage: set decimal digits of output score (_ts) by round method: 4 round-off and 5 round-up
                      default= 0, that means out score type is int

    :return: model: instance of model class, subclass of ScoreTransformModel
    """

    # check model name
    name = name.lower()
    if name.lower() not in mcfg.Models.keys():
        print('invalid name, not in {}'.format(list(mcfg.Models.keys())))
        return

    # check input data: DataFrame
    if type(df) != pd.DataFrame:
        if type(df) == pd.Series:
            df = pd.DataFrame(df)
        else:
            print('no score dataframe!')
            return
    else:
        df = df

    # check col
    if isinstance(cols, str):
        cols = [cols]
    elif type(cols) not in (list, tuple):
        print('invalid cols type:{}!'.format(type(cols)))
        return

    # check mode_ratio_prox
    if mode_ratio_prox not in ['lower_max', 'upper_min', 'near_min', 'near_max']:
        print('invalid approx mode: {}'.format(mode_ratio_prox))
        print('  valid approx mode: lower_max, upper_min, near_min, near_max')
        return
    if mode_ratio_cumu not in ['yes', 'no']:
        print('invalid cumu mode(yes/no): {}'.format(mode_ratio_cumu))
        return

    # ratio-seg score model: plt, ppt
    if (name in mcfg.Models.keys()) and (name not in ['tai', 'zscore', 'tscore']):
        ratio_tuple = tuple(x * 0.01 for x in mcfg.Models[name].ratio)
        plt_model = mbas.PltScore(name)
        plt_model.out_decimal_digits = 0
        plt_model.set_data(df=df, cols=cols)
        plt_model.set_para(
            raw_score_ratio_tuple=ratio_tuple,
            out_score_seg_tuple=mcfg.Models[name].section,
            raw_score_defined_max=max(raw_score_range),
            raw_score_defined_min=min(raw_score_range),
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            out_decimal_digits=out_decimal_digits
            )
        plt_model.run()
        return plt_model
    else:
        print('use run_model cols={}... '.format(cols))
        result = run_model(model_name=name,
                           df=df,
                           cols=cols,
                           raw_score_max=max(raw_score_range),
                           raw_score_min=min(raw_score_range),
                           mode_ratio_prox=mode_ratio_prox,
                           mode_ratio_cumu=mode_ratio_cumu,
                           mode_sort_order=mode_sort_order,
                           mode_section_point_first=mode_section_point_first,
                           mode_section_point_start=mode_section_point_start,
                           mode_section_point_last=mode_section_point_last,
                           mode_section_degraded=mode_section_degraded,
                           out_score_decimal=out_decimal_digits
                           )
        return result


# with model_name to get stm score by calling modelfunc.ModelAlgorithm
def run_model(
        model_name='shandong',
        df=None,
        cols=(),
        raw_score_max=100,
        raw_score_min=0,
        raw_score_step=1,
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        out_score_decimal=0,
        ):
    return mlib.ModelAlgorithm.get_stm_score(
        df=df,
        cols=cols,
        model_ratio_pdf=mcfg.Models[model_name].ratio,
        model_section=mcfg.Models[model_name].section,
        model_type=mcfg.Models[model_name].type.lower(),
        raw_score_max=raw_score_max,
        raw_score_min=raw_score_min,
        raw_score_step=raw_score_step,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_ratio_prox=mode_ratio_prox,
        mode_sort_order=mode_sort_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        out_score_decimal=out_score_decimal,
        )


# run to get stm score by calling methods in modelapi.ModelAlgorithm
def run_fun(
        df,
        cols,
        model_ratio_pdf,
        model_section,
        model_type='plt',
        raw_score_max=100,
        raw_score_min=0,
        raw_score_step=1,
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        out_score_decimal=0,
        ):
    return mlib.ModelAlgorithm.get_stm_score(
        df=df,
        cols=cols,
        model_ratio_pdf=model_ratio_pdf,
        model_section=model_section,
        model_type=model_type,
        raw_score_max=raw_score_max,
        raw_score_min=raw_score_min,
        raw_score_step=raw_score_step,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_ratio_prox=mode_ratio_prox,
        mode_sort_order=mode_sort_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        out_score_decimal=out_score_decimal,
        )
