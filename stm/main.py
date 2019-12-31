# coding: utf-8


"""
about this module:
    (1) function: run
    run() is an interface to call stmlib or stmlib2
    return model(stmlib.ScoreTransform) or result(stmlib2.get_stm_score.Result: (map_table, df))
    if model_name in ['zhejiang', 'shanghai', , 'beijing', 'tianjin',
                      'shandong', 'guangdong', 'SS7', 'hn900', 'hn300',
                      'hn300plt1', 'hn300plt2', 'hn300plt3']
        use stmlib.PltScore
    then
        call stmlib2.get_stm_score
    (2) function: run_model
    run_model() is an interface to call stmlib2 with model_name, df, cols
        and other parameters(score range, strategies, out score decimal digits)
    (3) function: run_para
    run_para() is an interface to call stmlib2 with df, cols, model_ratio, model_section, model_type,
        and other parameters(score range, strategies, out score decimal digits)

How to add new model in modelext:
    you can add a new model in modelext.Models_ext,
    then use the new model by calling run(model_name=new_model_name,...)
    in order to add new model in modelext.Models_ext,
    you can open the module modelext, modify Models_ext, add key-value: name-ModelFields
    then call run() with reload=True: result = run(model_name=new_model_name, ..., reload=True)
    then run() will check new model and add models in models_ext.Models to models_in.Models
    at last run() use new_model_name to call stmlib or stmlib2
"""


import sys
from stm import stmlib as slib, stmutil as utl, \
     stmlib2 as slib2, models_in as mdin, models_ext as mdext


def exp(name='shandong'):
    td = utl.TestData()()
    return run(model_name=name, df=td, cols=['km1'])


def run(
        model_name='shandong',
        df=None,
        cols=(),
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='descending',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        raw_score_range=(0, 100),
        out_score_decimals=0,
        reload=False,
        ):
    """

    [functions]
    run(model_name='shandong',  # model name in models_in.Models or models_ext.Models
        df=None,
        cols=None,
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='descending',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        raw_score_range=(0, 100),
        out_score_decimals=0,
        reload=False,
        )
        8个算法策略：
        --
        mode_ratio_prox: the mode to proxmate ratio value of raw score points
                         搜索对应比例的确定等级区间分值点的方式
              'upper_min': 小于该比例值的分值中最大的值   get score with min value in bigger percentile
              'lower_max': 大于该比例值的分值中最小的值   get score with max value in less percentile
               'near_min': 最接近该比例值的分值中最小的值 get score with min value in near percentile
               'near_max': 最接近该比例值的分值中最大的值 get score with max value in near percentile

          mode_ratio_cumu: 比例累加控制方式 use or not cumulative section ratio to locate section point
                    'yes': 以区间比例累计方式搜索 look up ratio with cumulative ratio
                     'no': 以区间比例独立方式搜索 look up ratio with interval ratio respectively

          ---
          usage:调用方式
          [1] from stm import main
          [2] m = main.run(name='shandong', df=data, col=['ls'])
          [3] m.report()
          [4] m.map_table.head()
          [5] m.outdf.head()
          [6] m.save_outdf_to_csv(filename_outdf)
          [7] m.save_map_table_doc(filename_maptable)

    [function] run()
    calling stmlib if model_type in 'plt, ppt' and calling stmlib2 if model_type is 'pgt'

    :param model_name: str, in models_in.Models.keys or models_ext.Models.keys
         values: 'shanghai', 'zhejiang', 'beijing', 'tianjin',  # plt model for grade score
                 'shandong', 'guangdong', 'SS7',                # plt model for linear mapping score
                 'hn900', 'hn300',                              # ppt model to transform score from (0,100)-->(60, 300)
                 'hn300plt1', 'hn300plt2', 'hn300plt3'          # plt model to transform score from (0,100)-->(60, 300)
                 'zscore', 'tscore'                             # ppt model for z, t, t-linear transform score
                 'tai'                                          # pgt model for taiwan grade score model
         default = 'shandong'
    :param df: pandas.DataFrame,
       values: raw score data, including score field, which type must be int or float
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
                            default= 'real'
    :param mode_section_point_start: str,
                           strategy: how to set first point of first section
                             values: 'real', use real raw score max or min value
                                     'defined', use test paper full score or least score
                            default= 'real'
    :param mode_section_point_last: str,
                           strategy: how to set first point of first section
                             values: 'real', use real raw score max or min value
                                     'defined', use test paper full score or least score
                            default= 'real'
    :param mode_section_lost: str,
                           strategy: how to prosess lost section
                             values: 'ignore', use real raw score max or min value
                                     'next_one_point',
                                     'next_two_point',
                            default= 'ignore'
    :param raw_score_range: tuple,
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= (0, 100)
    :param out_score_decimals: int, >=0
                        usage: set decimal digits of output score (_ts) by round method: 4 round-off and 5 round-up
                      default= 0, that means out score type is int
    :param reload: bool
            usage: reload related modules, especially when modify models_ext.Models
          default= False

    :return: (1) instance of PltScore, subclass of ScoreTransformModel, if 'plt' or 'ppt'
             (2) namedtuple('Model', ('outdf', 'map_table') if 'pgt'
    """

    if not check_for_run(
            model_name=model_name,
            df=df,
            cols=cols,
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            raw_score_range=raw_score_range,
            out_score_decimal_digits=out_score_decimals,
            reload=reload
            ):
        return None

    model_type = mdin.Models[model_name].type
    # model: plt, ppt
    if model_type in ['plt']:
        ratio_tuple = tuple(x * 0.01 for x in mdin.Models[model_name].ratio)
        plt_model = slib.PltScore(model_name, model_type)
        plt_model.set_data(df=df, cols=cols)
        plt_model.set_para(
            raw_score_ratio_tuple=ratio_tuple,
            out_score_seg_tuple=mdin.Models[model_name].section,
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
            out_decimal_digits=out_score_decimals
            )
        plt_model.run()
        return plt_model
    # for 'ppt', 'pgt' to call stmlib.Algorithm.get_stm_score
    else:
        print('run model by stmlib2, cols={} ... '.format(cols))
        result = run_model(
                           model_name=model_name,
                           df=df,
                           cols=cols,
                           raw_score_range=raw_score_range,
                           mode_ratio_prox=mode_ratio_prox,
                           mode_ratio_cumu=mode_ratio_cumu,
                           mode_sort_order=mode_sort_order,
                           mode_section_point_first=mode_section_point_first,
                           mode_section_point_start=mode_section_point_start,
                           mode_section_point_last=mode_section_point_last,
                           mode_section_degraded=mode_section_degraded,
                           out_score_decimals=out_score_decimals
                           )
        return result


# get stm score by calling stmlib2.ModelAlgorithm
def run_model(
        model_name='shandong',
        df=None,
        cols=(),
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        raw_score_range=(0, 100),
        raw_score_step=1,
        out_score_decimals=0,
        reload=False
        ):
    """
    to calculate out score by calling stmlib2.Algorithm.get_stm_score
    model_name in models_in.Models or models_ext.Models

    :param model_name:
    :param df:
    :param cols:
    :param raw_score_max:
    :param raw_score_min:
    :param raw_score_step:
    :param mode_ratio_cumu:
    :param mode_ratio_prox:
    :param mode_sort_order:
    :param mode_section_point_first:
    :param mode_section_point_start:
    :param mode_section_point_last:
    :param mode_section_degraded:
    :param mode_section_lost:
    :param out_score_decimals:
    :return:
    """
    if not check_for_run(
            model_name=model_name,
            df=df,
            cols=cols,
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            raw_score_range=raw_score_range,
            out_score_decimal_digits=out_score_decimals,
            reload=reload
            ):
        return None
    if model_name in mdin.Models.keys():
        model = mdin.Models[model_name]
    elif model_name in mdext.Models.keys():
        if utl.check_model(mdext.Models):
            model = mdext.Models[model_name]
        else:
            return None
    else:
        print('error model: {} is not in models_in.Models or modelext.Models_ext!'.format(model_name))
        return None

    return slib2.ModelAlgorithm.get_stm_score(
        df=df,
        cols=cols,
        model_ratio_pdf=model.ratio,
        model_section=model.section,
        model_type=model.type.lower(),
        raw_score_max=max(raw_score_range),
        raw_score_min=min(raw_score_range),
        raw_score_step=raw_score_step,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_ratio_prox=mode_ratio_prox,
        mode_sort_order=mode_sort_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        out_score_decimals=out_score_decimals,
        )


# calc stm score by calling methods in stmlib2.ModelAlgorithm
def run_para(
        df,
        cols,
        model_type=None,
        model_ratio_pdf=None,
        model_section=None,
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
        out_score_decimal_digits=0,
        ):
    """
    use each parameter to calculate out score by calling stmlib2

    :param df:
    :param cols:
    :param model_ratio_pdf:
    :param model_section:
    :param model_type:
    :param raw_score_max:
    :param raw_score_min:
    :param raw_score_step:
    :param mode_ratio_cumu:
    :param mode_ratio_prox:
    :param mode_sort_order:
    :param mode_section_point_first:
    :param mode_section_point_start:
    :param mode_section_point_last:
    :param mode_section_degraded:
    :param mode_section_lost:
    :param out_score_decimal_digits:
    :return:
    """
    return slib2.ModelAlgorithm.get_stm_score(
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
        out_score_decimals=out_score_decimal_digits,
        )


def check_for_run(
        model_name='shandong',
        df=None,
        cols=None,
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='descending',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        raw_score_range=(0, 100),
        out_score_decimal_digits=0,
        reload=False,
        ):

    # reload modules if any chenges done, especially in models_in.Models
    # reload modules: [x for x in sys.modules if 'stm' in x]
    if reload:
        print('reload modules ...')
        exec('import importlib as pb')
        for n1, n2, n3 in [('stm', 'slib',  'stmlib'),  ('stm', 'utl', 'stmutil'),
                           ('stm', 'slib2', 'stmlib2'), ('stm', 'mdin', 'models_in'),
                           ('stm', 'mdext',  'models_ext')]:
            if n1+'.'+n3 in sys.modules:
                exec('pb.reload('+n2+')')
        # check model in models_in
        for mk in mdin.Models.keys():
            if not utl.check_model(model_name=mk, model_lib=mdin.Models):
                print('error model: model={} in models_in.Models!'.format(mk))
                return False
        # add Models_ext to Models
        for mk in mdext.Models.keys():
            if not utl.check_model(model_name=mk, model_lib=mdext.Models):
                print('error model: model={} in models_ext.Models!'.format(mk))
                return False
            mdin.Models.update({mk: mdext.Models[mk]})

    # check model name
    model_name = model_name.lower()
    if model_name.lower() not in mdin.Models.keys():
        print('error name: name={} not in models_in.Models and models_ext.Models!'.format(model_name))
        return False

    # check input data: DataFrame
    if not utl.check_run_df_cols(df, cols, raw_score_range):
        return False

    # check strategy
    if not utl.check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
    ):
        return False

    if out_score_decimal_digits < 0 or out_score_decimal_digits > 10:
        print('warning: decimal digits={} set may error!'.format(out_score_decimal_digits))

    return True
