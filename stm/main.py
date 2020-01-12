# coding: utf-8


"""
about this module:
    (1) function: run
    run() is an interface to call stm1 or stm2
    return model(stmlib.ScoreTransform) or result(stm2.get_stm_score.Result: (map_table, df))
    if model_name_in ['zhejiang', 'shanghai', , 'beijing', 'tianjin',
                      'shandong', 'guangdong', 'SS7', 'hn900', 'hn300',
                      'hn300plt1', 'hn300plt2', 'hn300plt3'
                      ]
        use stm1.PltScore
    then
        call stm2.get_stm_score

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
    at last run() use new_model_name to call stm1 or stm2
"""


import time
import logging
from logging import handlers
import pandas as pd
import numbers

from stm import stm1, stm2, models_sys as mdsys, models_ext as mdext
from stm import main_config
import importlib as pb
stm_modules = [stm1, stm2, mdsys, mdext, main_config]


def run(model_name=None, df=None, cols=None):
    pb.reload(main_config)
    if model_name is None:
        model_name = main_config.model_name
    if (df is None) and (main_config.df is not None):
        df = main_config.df
    if (cols is None) and (main_config.cols is not None):
        cols = main_config.cols
    return runm(
        model_name=model_name,
        df=df,
        cols=cols,
        raw_score_range=main_config.run_parameters['raw_score_range'],
        mode_ratio_prox=main_config.run_strategy['mode_ratio_prox'],
        mode_ratio_cumu=main_config.run_strategy['mode_ratio_cumu'],
        mode_sort_order=main_config.run_strategy['mode_sort_order'],
        mode_section_point_first=main_config.run_strategy['mode_section_point_first'],
        mode_section_point_start=main_config.run_strategy['mode_section_point_start'],
        mode_section_point_last=main_config.run_strategy['mode_section_point_last'],
        mode_section_degraded=main_config.run_strategy['mode_section_degraded'],
        mode_section_lost=main_config.run_strategy['mode_section_lost'],
        mode_score_zero=main_config.run_strategy['mode_score_zero'],
        display=main_config.run_parameters['display'],
        logging=main_config.run_parameters['logging'],
        verify=main_config.run_parameters['verify'],
        out_score_decimals=main_config.run_parameters['out_score_decimals'],
        tiny_value=main_config.run_parameters['tiny_value'],
        )


def runm(
        model_name='shandong',
        df=None,
        cols=(),
        raw_score_range=(0, 100),
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        mode_score_zero='real',
        display=True,
        logging=None,
        verify=False,
        out_score_decimals=0,
        tiny_value=10**-8,
        ):

    """
    [functions]
    runm(model_name='shandong',  # model name in models_in.Models or models_ext.Models
         df=None,
         cols=None,
         raw_score_range=(0, 100),
         mode_ratio_prox='upper_min',
         mode_ratio_cumu='no',
         mode_sort_order='d',
         mode_section_point_first='real',
         mode_section_point_start='step',
         mode_section_point_last='real',
         mode_section_degraded='to_max',
         mode_section_lost='ignore',
         verify=False，
         logging=None,
         out_score_decimals=0,
         tiny_value=10**-8,
         )

        9个算法策略：
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
                    values: 'd', 'a'
                   default= 'd'

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
                             values: 'real', retain lost, use [-1, -1]
                                     'zip',  to next section, no [-1, -1] in middle
                            default= 'real'

    :param raw_score_range: tuple,
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= (0, 100)

    :param out_score_decimals: int, >=0
                        usage: set decimal digits of output score (_ts) by round method: 4 round-off and 5 round-up
                      default= 0, that means out score type is int

    :param verify: bool
            usage: use two algorithm to verify result
          default: False, do not verify

    :param tiny_value: float
                usage: control precision or equal
              default: 10**-8

    :param logging: bool
             usage: use logging to display messages to consol or write messages to file
           default: None

    :return: (1) instance of PltScore, subclass of ScoreTransformModel, if 'plt' or 'ppt'
             (2) namedtuple('Model', ('outdf', 'map_table') if 'pgt'
    """

    if logging:
        stmlogger = get_logger(model_name)
        stmlogger.loginfo_start('model: ' + model_name)
        if main_config.run_parameters['display']:
            stmlogger.logging_consol = True
        stmlogger.logging_file = True
    else:
        stmlogger = get_logger('test')
        stmlogger.logging_consol = True
        stmlogger.logging_file = False

    if not reload_stm_modules(stmlogger):
        stmlogger.loginfo_end('model: ' + model_name)
        return None

    if not check_run_parameters(
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
            mode_score_zero=mode_score_zero,
            raw_score_range=raw_score_range,
            out_score_decimal_digits=out_score_decimals,
            logger=stmlogger,
            ):
        stmlogger.loginfo_end('model: ' + model_name)
        return None

    model_type = mdsys.Models[model_name].type
    # model: plt, ppt
    if model_type in ['plt', 'ppt']:
        m1 = run1(
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
            mode_section_lost=mode_section_lost,
            mode_score_zero=mode_score_zero,    # not implemented in stm1
            out_score_decimals=out_score_decimals,
            display=display,
            tiny_value=tiny_value,
            logger=stmlogger,
            )
        result = m1
        if verify:
            verify_pass = True
            m2 = run2(
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
                mode_section_lost=mode_section_lost,
                mode_score_zero=mode_score_zero,
                out_score_decimals=out_score_decimals,
                display=display,
                tiny_value=tiny_value,
                logger=stmlogger,
                )
            for col in cols:
                out1 = m1.outdf.sort_values(col)[[col, col+'_ts']].values
                out2 = m2.outdf.sort_values(col)[[col, col+'_ts']].values
                comp = [(x, y) for x, y in zip(out1, out2) if x[1] != y[1]]
                if len(comp) > 0:
                    stmlogger.loginfo('verify error: col={},  {} records different in both algorithm!'.
                                      format(col, len(comp)))
                    for i in range(min(len(comp), 5)):
                        vs = 'stm1: {0} --> {1},   stm2: {2} -- > {3}'.format(*comp[i][0], *comp[i][1])
                        stmlogger.loginfo(vs)
                    if len(comp) > 5:
                        stmlogger.loginfo('...')
                    verify_pass = False
            if verify_pass:
                stmlogger.loginfo('verify passed!')
            result = (m1, m2)
    # 'pgt' to call stmlib.Algorithm.get_stm_score
    else:
        stmlogger.loginfo('run model by stmlib2, cols={} ... '.format(cols))
        result = run2(
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
                      mode_score_zero=mode_score_zero,
                      out_score_decimals=out_score_decimals,
                      display=display,
                      tiny_value=tiny_value,
                      logger=stmlogger,
                      )

    stmlogger.loginfo_end('model: ' + model_name)

    return result
    # end runm


def run1(
        model_name='shandong',
        df=None,
        cols=(),
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        mode_score_zero='real',
        raw_score_range=(0, 100),
        out_score_decimals=0,
        display=True,
        tiny_value=10 ** -8,
        logger=None,
        ):

    ratio_tuple = tuple(x * 0.01 for x in mdsys.Models[model_name].ratio)
    model_type = mdsys.Models[model_name].type
    m = stm1.PltScore(model_name, model_type)
    m.set_data(df=df, cols=cols)
    m.set_para(
        raw_score_ratio_tuple=ratio_tuple,
        out_score_seg_tuple=mdsys.Models[model_name].section,
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
        mode_score_zero=mode_score_zero,
        out_decimal_digits=out_score_decimals,
        display=display,
        tiny_value=tiny_value,
        logger=logger,
        )
    m.run()
    return m


# get stm score by calling stmlib2.ModelAlgorithm
def run2(
        model_name='shandong',
        df=None,
        cols=(),
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        mode_score_zero='real',
        raw_score_range=(0, 100),
        raw_score_step=1,
        out_score_decimals=0,
        display=True,
        tiny_value=10**-8,
        logger=None,
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

    if not check_run_parameters(
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
            logger=logger,
            ):
        return None

    model = mdsys.Models[model_name]
    return stm2.ModelAlgorithm.get_stm_score(
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
        mode_score_zero=mode_score_zero,
        out_score_decimals=out_score_decimals,
        display=display,
        tiny_value=tiny_value,
        logger=logger,
        )


# calc stm score by calling methods in stmlib2.ModelAlgorithm
def run2_para(
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
        mode_section_degraded='to_max',
        mode_section_lost='real',
        mode_score_zero='real',
        out_score_decimal_digits=0,
        display=True,
        logger=None,
        ):
    """
    use model parameters to call stmlib2.Algorithm.get_stm_score

    :param df:
    :param cols:
    :param model_type:
    :param model_ratio_pdf:
    :param model_section:
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
    :return: result(df, map_table),     df contains out score field [col+'_ts'] for con in cols
    """

    if not check_model_para(
        model_type=model_type,
        model_ratio=model_ratio_pdf,
        model_section=model_section,
        model_desc='',
        logger=logger,
        ):
        return None

    if not check_df_cols(
        df=df,
        cols=cols,
        raw_score_range=(raw_score_min, raw_score_max),
        logger=logger,
        ):
        return None

    # check strategy
    if not check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            mode_score_zero=mode_score_zero,
            logger=logger,
            ):
        return None

    return stm2.ModelAlgorithm.get_stm_score(
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
        mode_score_zero=mode_score_zero,
        out_score_decimals=out_score_decimal_digits,
        display=display,
        logger=logger,
        )
    # end--run2


def check_run_parameters(
        model_name='shandong',
        df=None,
        cols=None,
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='real',
        mode_score_zero='real',
        raw_score_range=(0, 100),
        out_score_decimal_digits=0,
        logger=None,
        ):

    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False

    if not check_merge_models():
        logger.loginfo('error: check models fail!')
        return False

    # check model name
    if model_name.lower() not in mdsys.Models.keys():
        logger.loginfo('error name: name={} not in models_in.Models and models_ext.Models!'.format(model_name))
        return False

    # check input data: DataFrame
    if not check_df_cols(df, cols, raw_score_range, logger):
        return False

    # check strategy
    if not check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            mode_score_zero=mode_score_zero,
            logger=logger,
    ):
        return False

    if out_score_decimal_digits < 0 or out_score_decimal_digits > 10:
        logger.logger.info('warning: decimal digits={} set may error!'.format(out_score_decimal_digits))

    return True


def reload_stm_modules(logger=None):
    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False
    logger.loginfo('reload modules ...')
    try:
        for m in stm_modules:
            pb.reload(m)
    except:
        return False
    return True


def check_merge_models(logger=None):
    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False
    # check model in models_in
    for mk in mdsys.Models.keys():
        if not check_model(model_name=mk, model_lib=mdsys.Models):
            logger.logger.info('error model: model={} in models_in.Models!'.format(mk))
            return False
    # add Models_ext to Models
    for mk in mdext.Models.keys():
        if not check_model(model_name=mk, model_lib=mdext.Models):
            logger.loginfo('error model: model={} in models_ext.Models!'.format(mk))
            return False
        mdsys.Models.update({mk: mdext.Models[mk]})
    return True


def check_model(model_name, model_lib=mdsys.Models, logger=None):
    if model_name in model_lib.keys():
        if not check_model_para(
            model_lib[model_name].type,
            model_lib[model_name].ratio,
            model_lib[model_name].section,
            model_lib[model_name].desc,
            logger=logger
        ):
            return False
    else:
        return False
    return True


def check_model_para(
                model_type='plt',
                model_ratio=None,
                model_section=None,
                model_desc='',
                logger=None,
                ):
    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False

    # check type
    if model_type not in ['ppt', 'plt', 'pgt']:
        logger.info('error type: valid type must be in {}'.format(model_type, ['ppt', 'plt', 'pgt']))
        return False

    # check ratio
    if model_type == 'pgt':
        if len(model_ratio) == 0:
            logger.loginfo('error ratio: length == 0 in model={}!'.format(model_type))
            return False
        if model_ratio[0] < 0 or model_ratio[0] > 100:
            logger.loginfo('error ratio: in type=tai, ratrio[0]={} must be range(0, 101) as the percent of top score ratio!'.format(model_ratio[0]))
            return False
    else:
        if len(model_ratio) != len(model_section):
            logger.loginfo('error length: the length of ratio group is not same as section group length !')
            return False
        if abs(sum(model_ratio) - 100) > 10**-12:
            logger.loginfo('error ratio: the sum of ratio must be 100, real sum={}!'.format(sum(model_ratio)))
            return False

    # check section
    for s in model_section:
        if len(s) > 2:
            logger.loginfo('error section: section must have 2 endpoints, real value: {}'.format(s))
            return False
        if s[0] < s[1]:
            logger.loginfo('error order: section endpoint order must be from large to small, '
                  'there: p1({}) < p2({})'.format(s[0], s[1]))
            return False
    if model_type in ['ppt', 'pgt']:
        if not all([x == y for x, y in model_section]):
            logger.loginfo('error section: ppt section, two endpoints must be same value!')
            return False

    # check desc
    if not isinstance(model_desc, str):
        logger.loginfo('error desc: model desc(ription) must be str, but real type={}'.format(type(model_desc)))

    return True


def check_strategy(
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='descending',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='map_to_max',
        mode_section_lost='ignore',
        mode_score_zero='real',
        logger=None
        ):

    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False

    st = {'mode_ratio_prox': mode_ratio_prox,
          'mode_ratio_cumu':mode_ratio_cumu,
          'mode_sort_order': mode_sort_order,
          'mode_section_point_first': mode_section_point_first,
          'mode_section_point_start': mode_section_point_start,
          'mode_section_point_last': mode_section_point_last,
          'mode_section_degraded': mode_section_degraded,
          'mode_section_lost': mode_section_lost,
          'mode_score_zero': mode_score_zero,
          }
    for sk in st.keys():
        if sk in mdsys.Strategy.keys():
            if not st[sk] in mdsys.Strategy[sk]:
                logger.loginfo('error mode: {}={} not in {}'.format(sk, st[sk], mdsys.Strategy[sk]))
                return False
        else:
            logger.loginfo('error mode: {} is not in Strategy-dict!'.format(sk))
            return False
    return True


def check_df_cols(df=None, cols=None, raw_score_range=None, logger=None):
    if logger is None:
        logger = get_logger('test')
        logger.logging_consol = True
        logger.logging_file = False
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, pd.Series):
            logger.loginfo('warning: df is pandas.Series!')
            return False
        else:
            logger.loginfo(logger, 'error data: df is not pandas.DataFrame!')
            return False
    if len(df) == 0:
        logger.loginfo('error data: df is empty!')
        return False
    if type(cols) not in (list, tuple):
        logger.loginfo('error type: cols must be list or tuple, real type is {}!'.format(type(cols)))
        return False
    for col in cols:
        if type(col) is not str:
            logger.loginfo('error col: {} is not str!'.format(col))
            return False
        else:
            if col not in df.columns:
                logger.loginfo('error col: {} is not in df.columns!'.format(col))
                return False
            if not isinstance(df[col][0], numbers.Real):
                logger.loginfo('type error: column[{}] not Number type!'.format(col))
                return False
            _min = df[col].min()
            if _min < min(raw_score_range):
                logger.loginfo('warning: some scores in col={} not in raw score range:{}'.format(_min, raw_score_range))
            _max = df[col].max()
            if _max > max(raw_score_range):
                logger.loginfo('warning: some scores in col={} not in raw score range:{}'.format(_max, raw_score_range))
    return True


def get_logger(model_name):
    gmt = time.localtime()
    log_file = model_name + str(gmt.tm_year) + str(gmt.tm_mon) + str(gmt.tm_mday) + '.log'
    stmlog = Logger(log_file, level='info')
    return stmlog


class Logger(object):
    level_relations = {
        'debug':    logging.DEBUG,
        'info':     logging.INFO,
        'warn':     logging.WARNING,
        'error':    logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
                 filename='stm.log',
                 level='info',
                 when='D',
                 back_count=3,
                 ):

        # stat
        self.logging_file = True
        self.logging_consol = True

        # para
        self.filename = filename
        self.level = level
        self.when = when
        self.back_count = 3 if back_count is not int else back_count
        self.format = '   %(message)s'
        self.when = when
        # self.format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        # set logger
        self.logger = logging.getLogger(self.filename)              # file name
        self.logger.setLevel(self.level_relations.get(self.level))  # 设置日志级别
        self.logger_format = logging.Formatter(self.format)         # 设置日志格式

        # set handlers
        self.stream_handler = None
        self.rotating_file_handler = None
        self.set_handlers(self.logger_format)

    def loginfo(self, ms=''):
        self.logger.handlers = []
        if self.logging_consol:
            self.logger.addHandler(self.stream_handler)
        if self.logging_file:
            self.logger.addHandler(self.rotating_file_handler)
        self.logger.info(ms)
        self.logger.handlers = []

    def loginfo_start(self, ms=''):
        first_logger_format = logging.Formatter('='*120 + '\n[%(message)s] start at [%(asctime)s]\n' + '-'*120)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        self.set_handlers(self.logger_format)

    def loginfo_end(self, ms=''):
        first_logger_format = logging.Formatter('-'*120 + '\n[%(message)s]  end at [%(asctime)s]\n' + '='*120)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        self.set_handlers(self.logger_format)

    def set_handlers(self, log_format):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(log_format)
        self.rotating_file_handler = handlers.TimedRotatingFileHandler(
                    filename=self.filename,
                    when=self.when,
                    backupCount=self.back_count,
                    encoding='utf-8'
                )
        self.rotating_file_handler.setFormatter(log_format)
