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
    you can add a new model in models_ext.Models_ext,
    then use the new model by calling runc(model_name=new_model_name,...) or runm(...)
    in order to add new model in modelext.Models_ext,
    open modeles_ext.py, modify Models_ext, add key-value: name-ModelFields
    then call run() with reload=True: result = run(model_name=new_model_name, ..., reload=True)
    then run() will check new model and add models in models_ext.Models_ext to models_in.Models
    at last run() use new_model_name to call stm1 or stm2
"""

from collections import namedtuple

from stm import stmlib, stm1, stm2, models
import importlib as pb
stm_modules = [stmlib, stm1, stm2, models]


def run_config(conf_name='stm.conf'):

    for m in stm_modules:
        pb.reload(m)

    config_read = stmlib.read_conf(conf_name)
    if config_read:
        mcfg = config_read
    else:
        print('read config file {} fail!'.format(conf_name))
        return None

    # no model from built-in models
    if not mcfg['model_in_check']:
        # new model in config-file
        if mcfg['model_new_check']:
            mcfg.update({'model_name': mcfg['model_new_name']})
            models.Models.update(
                {mcfg['model_name']:
                models.ModelFields(
                    mcfg['model_new_type'],
                    mcfg['model_new_ratio'],
                    mcfg['model_new_section'],
                    mcfg['model_new_desc']
                )}
            )
            print('new model [{}] check-in!'.format(mcfg['model_new_name']))
        # no model can be used
        else:
            print('new model check-in fail! no model can be used!')
            return mcfg

    if 'df' in mcfg.keys():
        if mcfg['df'] is None:
            print('data not found!')
            return mcfg
    if 'cols' in mcfg.keys():
        if mcfg['cols'] is None:
            print('cols not found!')
            return mcfg

    for k in mcfg.keys():
        if k == 'df':
            print('{:25s}: {:10s}'.format(k, str(mcfg[k].columns)))
        else:
            print('{:25s}: {:10s}'.format(k, str(mcfg[k])))

    result = run(
        model_name=mcfg['model_name'],
        df=mcfg['df'],
        cols=mcfg['cols'],
        raw_score_min=mcfg['raw_score_min'],
        raw_score_max=mcfg['raw_score_max'],
        mode_ratio_prox=mcfg['mode_ratio_prox'],
        mode_ratio_cumu=mcfg['mode_ratio_cumu'],
        mode_sort_order=mcfg['mode_sort_order'],
        mode_section_point_first=mcfg['mode_section_point_first'],
        mode_section_point_start=mcfg['mode_section_point_start'],
        mode_section_point_last=mcfg['mode_section_point_last'],
        mode_section_degraded=mcfg['mode_section_degraded'],
        mode_section_lost=mcfg['mode_section_lost'],
        logname=mcfg['logname'],
        logdisp=mcfg['logdisp'],
        logfile=mcfg['logfile'],
        verify=mcfg['verify'],
        out_score_decimals=mcfg['out_score_decimals'],
        tiny_value=mcfg['tiny_value'],
        )

    if not result.ok:
        print('run fail: result is None!')
        # return mcfg

    return result


def run(
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
        logname=None,
        logdisp=True,
        logfile=False,
        verify=False,
        raw_score_min=0,
        raw_score_max=100,
        out_score_decimals=0,
        tiny_value=10**-12,
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

    :param model_name: str, in models_in.Models.keys or models_ext.Models_ext.keys
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

    :param mode_score_zero：str
                    strategy: how to prosess lost section
                      values: 'real',   retain real zero percent to map
                              'after',  transform zero score to min after stm with zero records
                              'alone',  transform zero alone, move zero score records from df
                     default= 'real'

    :param raw_score_min: int
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= 0
    :param raw_score_max: int
                     usage: raw score value range (min, max)
                    values: max and min raw score full and least value in paper
                   default= 100

    :param verify: bool
            usage: use two algorithm to verify result
          default: False, do not verify

    :param logdisp: bool
             usage: display messages to consol
           default: True

    :param logfile: bool
             usage: use logging to display messages to consol or write messages to file
           default: None

    :param out_score_decimals: int, >=0
                        usage: set decimal digits of output score (_ts) by round method: 4 round-off and 5 round-up
                      default= 0, that means out score type is int

    :param tiny_value: float
                usage: control precision or equal
              default: 10**-8

    :return: namedtuple(ok, r1, r2)
            (1) ok: bool, successful or not
            (2) r1: result of stm1, instance of PltScore, subclass of ScoreTransformModel
            (3) r2: result of stm2, namedtuple('Model', ('outdf', 'map_table')
    """
    raw_score_range = (raw_score_min, raw_score_max)

    result = namedtuple('Result', ['ok', 'r1', 'r2'])

    stmlogger = stmlib.get_logger(model_name, task=logname)
    stm_no = '  No.' + str(id(stmlogger))
    if logdisp:
        stmlogger.logging_consol = True
    if logfile:
        stmlogger.logging_file = True
    stmlogger.loginfo_start('model:' + model_name + stm_no)

    # if not stmlib.Checker.reload_stm_modules(stmlogger, stm_modules):
    #     stmlogger.loginfo('reload error: can not reload modules:{}'.format(stm_modules))
    #     stmlogger.loginfo_end('model:' + model_name + stm_no +
    #                           '  df.colums={} score fields={}\n'.format(list(df.columns), cols))
    #     return result(False, None, None)
    #
    # if not stmlib.Checker.check_merge_models(logger=stmlogger,
    #                                          sys_models=mdsys.Models,
    #                                          ext_models=mdext.Models_ext):
    #     # stmlogger.loginfo('error: models_sys-models_ext merge fail!')
    #     return False

    if not stmlib.Checker.check_run(
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
            logger=stmlogger,
            models=models,
            ):
        stmlogger.loginfo_end('model:' + model_name + stm_no)
        return result(False, None, None)
    stmlogger.loginfo('data columns: {}, score fields: {}'.format(list(df.columns), cols))

    model_type = models.Models[model_name].type
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
            out_score_decimals=out_score_decimals,
            tiny_value=tiny_value,
            logger=stmlogger,
            )
        r = result(True, m1, None)
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
                out_score_decimals=out_score_decimals,
                tiny_value=tiny_value,
                logger=stmlogger,
                )
            for col in cols:
                out1 = m1.outdf.sort_values(col)[[col, col+'_ts']].values
                out2 = m2.outdf.sort_values(col)[[col, col+'_ts']].values
                comp = [(x, y) for x, y in zip(out1, out2) if x[1] != y[1]]
                if len(comp) > 0:
                    stmlogger.loginfo('verify fail: col={},  {} records different in both algorithm!'.
                                      format(col, len(comp)))
                    for i in range(min(len(comp), 5)):
                        vs = 'stm1: {0} --> {1},   stm2: {2} -- > {3}'.format(*comp[i][0], *comp[i][1])
                        stmlogger.loginfo(vs)
                    if len(comp) > 5:
                        stmlogger.loginfo('...')
                    verify_pass = False
            if verify_pass:
                stmlogger.loginfo('verify passed!')
            r = result(verify_pass, m1, m2)
    # 'pgt' to call stmlib.Algorithm.get_stm_score
    else:
        stmlogger.loginfo('run model by stm2, cols={} ... '.format(cols))
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
                  out_score_decimals=out_score_decimals,
                  tiny_value=tiny_value,
                  logger=stmlogger,
                  )
        r = result(True, None, m2)

    stmlogger.loginfo('result data: {}\n    score cols: {}'.format(list(df.columns), cols))
    stmlogger.loginfo_end('model:{}{} '.format(model_name, stm_no))
    stmlogger.logger.handlers = []
    del stmlogger.logging_file
    del stmlogger

    return r
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
        raw_score_range=(0, 100),
        out_score_decimals=0,
        tiny_value=10 ** -12,
        logger=None,
        ):

    ratio_tuple = tuple(x * 0.01 for x in models.Models[model_name].ratio)
    model_type = models.Models[model_name].type
    m = stm1.PltScore(model_name, model_type)
    m.set_data(df=df, cols=cols)
    m.set_para(
        raw_score_ratio=ratio_tuple,
        out_score_section=models.Models[model_name].section,
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
        raw_score_range=(0, 100),
        raw_score_step=1,
        out_score_decimals=0,
        tiny_value=10**-12,
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

    if not stmlib.Checker.check_run(
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
            models=models.Models,
            ):
        return None

    model = models.Models[model_name]
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
        out_score_decimals=out_score_decimals,
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
        out_score_decimal_digits=0,
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
    :param logger: set logger
    :return: result(outdf, map_table),     out_score_field[col+'_ts'] for col in cols in outdf
    """

    if not stmlib.Checker.check_model_para(
                                    model_type=model_type,
                                    model_ratio=model_ratio_pdf,
                                    model_section=model_section,
                                    model_desc='',
                                    logger=logger,
                                    ):
        return None

    if not stmlib.Checker.check_df_cols(
                                df=df,
                                cols=cols,
                                raw_score_range=(raw_score_min, raw_score_max),
                                logger=logger,
                                ):
        return None

    # check strategy
    if not stmlib.Checker.check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            logger=logger,
            models=models
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
        out_score_decimals=out_score_decimal_digits,
        logger=logger,
        )
    # end--run2
