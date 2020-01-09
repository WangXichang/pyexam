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
from stm import stm1, stm2, stmlib as slib,\
     main_util as utl, models_in as mdin, models_ext as mdext
import importlib as pb
stm_modules = [stm1, stm2, utl, mdin, mdext]
from stm import main_config


def exp(name='shandong'):
    return run(model_name=name,
               df=utl.TestData()(),
               cols=['km1', 'km2'],
               reload=True)


def run(model_name=None, df=None, cols=None):
    pb.reload(main_config)

    if model_name is None:
        model_name = main_config.model_name
    if (df is None) and (main_config.df is not None):
        df = main_config.df
    if (cols is None) and (main_config.cols is not None):
        cols = main_config.cols

    stg = list(main_config.run_strategy.values())
    oth = list(main_config.run_other_para.values())
    pp = [model_name, df, cols] + stg + oth

    return runm(*pp)


def runm(
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
        reload=False,
        save_result=False,
        path_name=None,
        display=True,
        verify=False,
        tiny_value=10**-8,
        ):

    """

    [functions]
    run(model_name='shandong',  # model name in models_in.Models or models_ext.Models
        df=None,
        cols=None,
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_sort_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='ignore',
        raw_score_range=(0, 100),
        out_score_decimals=0,
        reload=False,
        verify=False
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
    :param reload: bool
            usage: reload related modules, especially when modify models_ext.Models
          default= False, dont reload

    :param verify: bool
            usage: use two algorithm to verify result
          default: False, do not verify

    :param tiny_value: float
                usage: control precision or equal
              default: 10**-8

    :return: (1) instance of PltScore, subclass of ScoreTransformModel, if 'plt' or 'ppt'
             (2) namedtuple('Model', ('outdf', 'map_table') if 'pgt'
    """

    if reload:
        if not reload_stm_modules():
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
            ):
        return None

    model_type = mdin.Models[model_name].type
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
            reload=False,
            display=display,
            tiny_value=tiny_value,
            )
        if verify:
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
                reload=False,
                display=display,
                tiny_value=tiny_value,
            )
            for col in cols:
                if not all(m1.outdf[col+'_ts'] == m2.outdf[col+'_ts']):
                    print('verify error: col={} get different result in both algorithm!'.format(col))
                    return m1, m2
            if display:
                print('verify passed!')
            return m1, m2
        return m1
    # 'pgt' to call stmlib.Algorithm.get_stm_score
    else:
        if display:
            print('run model by stmlib2, cols={} ... '.format(cols))
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
                       reload=False,
                       display=display,
                       tiny_value=tiny_value,
                       )
        if save_result:
            if isinstance(path_name, str):
                save_map_table(path_name, model_name, map_table=result.map_table)
                save_out_score(path_name, model_name, outdf=result.outdf)
            else:
                print('error path_name: {}'.format(path_name))
        return result


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
        reload=False,
        tiny_value=10 ** -8,
        ):

    if reload:
        if not reload_stm_modules():
            return None

    ratio_tuple = tuple(x * 0.01 for x in mdin.Models[model_name].ratio)
    model_type = mdin.Models[model_name].type
    m = stm1.PltScore(model_name, model_type)
    m.set_data(df=df, cols=cols)
    m.set_para(
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
        out_decimal_digits=out_score_decimals,
        display=display,
        tiny_value=tiny_value,
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
        reload=False,
        display=True,
        tiny_value=10**-8,
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

    # reload modules if any chenges done, especially in models_in.Models
    if reload:
        if not reload_stm_modules():
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
            raw_score_range=raw_score_range,
            out_score_decimal_digits=out_score_decimals,
            ):
        return None

    model = mdin.Models[model_name]
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
        reload=False,
        display=True,
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
    :param reload: reload lib, lib2,utl, models_in, models_ext
    :return: result(df, map_table),     df contains out score field [col+'_ts'] for con in cols
    """

    if reload:
        if not reload_stm_modules():
            return None

    if not utl.check_model_para(
        model_type=model_type,
        model_ratio=model_ratio_pdf,
        model_section=model_section,
        model_desc=''
        ):
        return None

    if not utl.check_df_cols(
        df=df,
        cols=cols,
        raw_score_range=(raw_score_min, raw_score_max)
        ):
        return None

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
            mode_score_zero=mode_score_zero,
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
        )


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
        ):

    if not check_merge_models():
        return False

    # check model name
    if model_name.lower() not in mdin.Models.keys():
        print('error name: name={} not in models_in.Models and models_ext.Models!'.format(model_name))
        return False

    # check input data: DataFrame
    if not utl.check_df_cols(df, cols, raw_score_range):
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
            mode_score_zero=mode_score_zero,
    ):
        return False

    if out_score_decimal_digits < 0 or out_score_decimal_digits > 10:
        print('warning: decimal digits={} set may error!'.format(out_score_decimal_digits))

    return True


def reload_stm_modules():
    print('reload modules ...')
    try:
        for m in stm_modules:
            pb.reload(m)
    except:
        return False
    return True


def check_merge_models():
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
    return True


def save_map_table(path_name=None, model_name=None, file_type='csv', map_table=None):
    """
    save map table to file
    """
    ts = time.asctime().replace(' ', '-')
    file_name = path_name + model_name + '_map_table_' + ts + '.' + file_type
    if map_table is not None:
        map_table.to_csv(file_name)


def save_out_score(path_name=None, model_name=None, file_type='csv', outdf=None):
    """
    save out score to file
    """
    ts = time.asctime().replace(' ', '-')
    file_name = path_name + model_name + '_out_score_' + ts + '.' + file_type
    if outdf is not None:
        outdf.to_csv(file_name)