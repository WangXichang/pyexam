# coding: utf-8


import pandas as pd
import sys
from stm import stmlib as mlib, modelutil as mutl, stmlib2 as mlib2, modelsetin as msin, modelext as mext


def exp(name='shandong'):
    td = mutl.TestData()()
    return run(name=name, df=td, cols='km1')


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
        out_decimal_digits=0,
        reload=True,
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
    :param mode_section_lost: str,
                           strategy: how to prosess lost section
                             values: 'ignore', use real raw score max or min value
                                     'next_one',
                                     'next_one',
                            default= 'ignore'
    :param raw_score_range: tuple,
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= (0, 100)
    :param out_decimal_digits: int,
                        usage: set decimal digits of output score (_ts) by round method: 4 round-off and 5 round-up
                      default= 0, that means out score type is int

    :return: model: instance of model class, subclass of ScoreTransformModel
    """

    # reload modules if any chenges done, especially in modelsetin.Models
    # reload modules: [x for x in sys.modules if 'stm' in x]
    if reload:
        print('reload modules ...')
        exec('import importlib as pb')
        for n1, n2, n3 in [('stm', 'mlib',  'stmlib'),  ('stm', 'mutl', 'modelutil'),
                           ('stm', 'mlib2', 'stmlib2'), ('stm', 'msin', 'modelsetin'),
                           ('stm', 'mext',  'modelext')]:
            if n1+'.'+n3 in sys.modules:
                exec('pb.reload('+n2+')')
        # add Models_ext to Models
        for mk in mext.Models_ext.keys():
            if not mutl.check_model(model_name=mk):
                print('error model: model={} defined incorrectly!'.format(mk))
                return False
            msin.Models.update({mk: mext.Models_ext[mk]})

    # check model name
    name = name.lower()
    if name.lower() not in msin.Models.keys():
        print('error name: name={} not in modelsetin.Models and modelext.Models_ext!'.format(name))
        return None

    # check input data: DataFrame
    if type(df) != pd.DataFrame:
        if type(df) == pd.Series:
            df = pd.DataFrame(df)
        else:
            print('error data: df is not a pandas.DataFrame or pandas.Series!')
            return

    # check col
    if isinstance(cols, str):
        cols = [cols]
    # condition is: sequence and element is str and is a column name in columns
    elif type(cols) not in (list, tuple):
        print('error type: cols must be list or tuple, real type is {}!'.format(type(cols)))
        return None

    # check col type
    import numbers
    if len(df) > 0:
        for col in cols:
            if col not in df.columns:
                print('error col: [{}] is not a name of df columns!'.format(col))
                return None
            if not isinstance(df[col][0], numbers.Number):
                print('type error: column[{}] not Number type!'.format(col))
                return None

    # check mode_ratio_prox
    if mode_ratio_prox not in ['lower_max', 'upper_min', 'near_min', 'near_max']:
        print('invalid approx mode: {}'.format(mode_ratio_prox))
        print('  valid approx mode: lower_max, upper_min, near_min, near_max')
        return
    if mode_ratio_cumu not in ['yes', 'no']:
        print('invalid cumu mode(yes/no): {}'.format(mode_ratio_cumu))
        return

    # ratio-seg score model: plt, ppt
    if (name in msin.Models.keys()) and (name not in ['tai', 'zscore', 'tscore']):
        ratio_tuple = tuple(x * 0.01 for x in msin.Models[name].ratio)
        plt_model = mlib.PltScore(name)
        plt_model.out_decimal_digits = 0
        plt_model.set_data(df=df, cols=cols)
        plt_model.set_para(
            raw_score_ratio_tuple=ratio_tuple,
            out_score_seg_tuple=msin.Models[name].section,
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
    if model_name in msin.Models.keys():
        model = msin.Models[model_name]
    elif model_name in mext.Models_ext.keys():
        if mutl.check_model(mext.Models_ext):
            model = mext.Models_ext[model_name]
        else:
            return None
    else:
        print('error model: {} is not in modelsetin.Models or modelext.Models_ext!'.format(model_name))
        return None

    return mlib2.ModelAlgorithm.get_stm_score(
        df=df,
        cols=cols,
        model_ratio_pdf=model.ratio,
        model_section=model.section,
        model_type=model.type.lower(),
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


# run to get stm score by calling methods in stmlib2.ModelAlgorithm
def run_lib2(
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
    return mlib2.ModelAlgorithm.get_stm_score(
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
