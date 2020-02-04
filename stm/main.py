# coding: utf-8


"""
about module main:
    (1) function: run
    run() is an interface to call stm1 or stm2
    return model(stmlib.ScoreTransform) or result(stm2.get_stm_score.Result: (map_table, df))
    if model_name_in ['zhejiang', 'shanghai', , 'beijing', 'tianjin',
                      'shandong', 'guangdong', 'p7', 'h900', 'hainan',
                      'hn300plt1', 'hn300plt2', 'hn300plt3'
                      ]
        call stm1.PltScore
    then
        call stm2.get_stm_score

    (2) function: run_conf, new_conf
    run_conf(confname) is an interface to call stm1/stm2 with model_name, df, cols
    and other parameters(score range, strategies, out score decimal digits)
    run_conf need to call a config file to set model parameters, the filename is confname
    you can use new_conf(confname) to create a config file with name=confname

    (3) function: run1
    run1 is an interface to call stm1 with model name, df, cols and other parameters

    (4) function: run2
    run2 is an interface to call stm2 with model name, df, cols and other parameters

    (5) function: run_para
    run_para() is an interface to call stm2 with df, cols, model_ratio, model_section, model_type,
    and other parameters(score range, strategies, out score decimal digits).
    used to test algorithm commonly.

"""


import time as __time
import os as __os
from collections import namedtuple as __namedtuple
import importlib as __pb

from stm import stmlib as __slib, stm1 as __stm1, stm2 as __stm2, modelset as __mdset
__stm_modules = [__slib, __stm1, __stm2, __mdset]


def new_conf(confname='stm001.conf'):
    if __slib.isfilestr(confname):
        __slib.make_config_file(confname)
    else:
        print('invalid file name!')


def runf(conf_name='stm.conf'):

    if not __os.path.isfile(conf_name):
        print('conf file: {} not found!'.format(conf_name))
        return None

    for m in __stm_modules:
        __pb.reload(m)

    config_read = __slib.read_conf(conf_name)
    if config_read:
        mcfg = config_read
    else:
        print('read config file {} fail!'.format(conf_name))
        return None

    # use new model when no model defined, that is in modelset
    if not mcfg['model_in_check']:
        # new model in config-file
        if mcfg['model_new_check']:
            mcfg.update({'model_name': mcfg['model_new_name']})
            __mdset.Models.update(
                {mcfg['model_name']:
                __mdset.ModelFields(
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

    if mcfg['df'] is None:
        return mcfg
    if mcfg['cols'] is None:
        return mcfg

    print('=' * 120)
    print('stm setting in {}'.format(conf_name) + '\n' + '-'*120)
    for k in mcfg.keys():
        if k == 'df':
            print('   {:25s}: {:10s}'.format(k, str(mcfg[k].columns)))
        else:
            print('   {:25s}: {:10s}'.format(k, str(mcfg[k])))
    print('=' * 120 + '\n')

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
        loglevel=mcfg['loglevel'],
        saveresult=mcfg['saveresult'],
        out_score_decimals=mcfg['out_score_decimals'],
        tiny_value=mcfg['tiny_value'],
        verify=mcfg['verify'],
        )

    if not result:
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
        loglevel='info',
        saveresult=True,
        verify=False,
        raw_score_min=0,
        raw_score_max=100,
        out_score_decimals=0,
        tiny_value=10**-12,
        ):

    """
    ---
    基本参数：
        model_name: str, 转换模型名称
        df: DataFrame, 原始分数数据
        cols: list of str, 需要转换的原始分数列（字段）
    ---
    算法策略参数：
          mode_ratio_prox: the mode to proxmate ratio value of raw score points
                         搜索对应比例的确定等级区间分值点的方式
              'upper_min': 小于该比例值的分值中最大的值   get score with min value in bigger percentile
              'lower_max': 大于该比例值的分值中最小的值   get score with max value in less percentile
               'near_min': 最接近该比例值的分值中最小的值 get score with min value in near percentile
               'near_max': 最接近该比例值的分值中最大的值 get score with max value in near percentile

          mode_sort_order: 分数排序方式 score sort method: descending or ascending
                      'd': 降序方式排序，从高分数开始计算
                      'a': 升序方式排序，从低分数开始计算

          mode_ratio_cumu: 比例累加控制方式 use or not cumulative section ratio to locate section point
                    'yes': 以区间比例累计方式搜索 look up ratio with cumulative ratio
                     'no': 以区间比例独立方式搜索 look up ratio with interval ratio respectively

          mode_section_point_first: 区间的第一个点的取值方式
                            'real': 取实际值，即实际得分的最高分（descending）或最低分数(ascending)
                         'defined': 取定义值，即卷面定义的最高分raw_score_min（descending）或最低分数raw_score_max(ascending)

          mode_section_point_first: 区间的第一个点的取值方式
                            'real': 取实际值，即实际得分的最高分（descending）或最低分数(ascending)
                         'defined': 取定义值，即卷面定义的最高分raw_score_min（descending）或最低分数raw_score_max(ascending)

          mode_section_point_start: 区间的开始点的取值方式（第一个区间开始点使用mode_section_point_first确定）
                            'step': 取顺延值，即上一个区间的末端点值加1（descending）或减1(ascending)
                           'share': 取共享值，即上一个区间的末端点值

          mode_section_point_last: 区间的末端点的取值方式
                            'real': 取实际值，即实际得分的最高分（descending）或最低分数(ascending)
                         'defined': 取定义值，即卷面定义的最高分raw_score_min（descending）或最低分数raw_score_max(ascending)

             mode_section_degraded: 区间退化处理方式
                          'to_max': 取对应输出区间的最大值
                          'to_min': 取对应输出区间的最小值
                         'to_mean': 取对应输出区间的平均值
    ---
    任务控制参数：
        logname: str, 任务名称， 用于日志文件、输出数据文件的前缀
        logdisp: bool, 是否显示计算运行结果
        logfile: bool, 是否将计算运行结果写入日志文件
        loglevel: str, 输出结果的等级：'debug', 'info'
        saveresult: bool, 是否将计算结果写入文件，包括转换输出分数(df_outscore)、转换映射表(df_maptable)
        verify: bool, 是否使用算法验证，即使用两种计算算法对计算结果进行验证
        ---
        分值数值与计算精度：
        raw_score_min: int, 原始分数的最小值
        raw_score_max: int, 原始分数的最大值
        out_score_decimals: int, 输出分数的小数位数
        tiny_value: float, 最小精度值，用于过程计算的精度控制， 一般可设为10**-10
    ---
    函数返回值
    返回值为含有三个带有名称（ok, r1, r2）的元素的元组  namedtuple(ok, r1, r2)
      (1) ok: bool, 计算是否成功 successful or not
      (2) r1: 主算法的计算结果
              模块stm1中类PltScore的实例
              主要数据结果是PltScore.outdf, PltScore.map_table
              result of stm1, instance of PltScore
      (3) r2: 辅助算法的计算结果， 模块stm2中函数ModelAlgorithm.get_stm_score()的返回结果，
              元素名称为outdf,map_table的元组
              如果不指定verify 或 verify != True时，r2为None
              result of stm2, namedtuple('outdf', 'map_table')
              r2 is None if verify != True
    ---
    调用方式
      [1] from stm import main
      [2] m = main.run(model_name='zhejiang', df=data, col=['ls'])
      [3] m.ok
      [4] m.r1.map_table.head()
      [5] m.r1.outdf.head()
    ---
    """

    raw_score_range = (raw_score_min, raw_score_max)

    if not logname:
        task = 'stm'
    else:
        task = logname

    result = __namedtuple('Result', ['ok', 'r1', 'r2'])
    r = result(False, None, None)

    stmlogger = __slib.get_logger(model_name, logname=task)
    stmlogger.set_level(loglevel)
    stm_no = '  No.' + str(id(stmlogger))
    if logdisp:
        stmlogger.logging_consol = True
    if logfile:
        stmlogger.logging_file = True

    stmlogger.loginfo('\n*** running begin ***')
    stmlogger.loginfo_start('task:' + task + ' model:' + model_name + stm_no)

    if not __slib.Checker.check_run(
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
            models=__mdset,
            ):
        stmlogger.loginfo_end('task:' + task + ' model:' + model_name + stm_no)
        return result(False, None, None)
    stmlogger.loginfo('data columns: {}, score fields: {}'.format(list(df.columns), cols))

    model_type = __mdset.Models[model_name].type
    # plt, ppt : call stm1.PltScore
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
    # 'pgt': call stmlib.Algorithm.get_stm_score
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

    if r.ok:
        t = __time.localtime()
        fno = '_'.join(map(str, [t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec]))
        save_dfscore_name = task + '_df_outscore_' + model_name + '_' + fno + '.csv'
        save_dfmap_name = task + '_df_maptable_' + model_name + '_' + fno + '.csv'
        if r.r1 is not None:
            dfscore = r.r1.outdf
            dfmaptable = r.r1.map_table
        else:
            dfscore = r.r2.outdf
            dfmaptable = r.r2.map_table
        if saveresult:
            dfscore.to_csv(save_dfscore_name, index=False)
            dfmaptable.to_csv(save_dfmap_name, index=False)
        stmlogger.loginfo('result data: {}\n    score cols: {}'.format(list(dfscore.columns), cols))
        stmlogger.loginfo_end('task:' + task + '  model:{}{} '.format(model_name, stm_no))
    else:
        stmlogger.loginfo_end('model={} running fail!'.format(model_name))

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

    ratio_tuple = tuple(x * 0.01 for x in __mdset.Models[model_name].ratio)
    model_type = __mdset.Models[model_name].type
    m = __stm1.PltScore(model_name, model_type)
    m.set_data(df=df, cols=cols)
    m.set_para(
        raw_score_ratio=ratio_tuple,
        out_score_section=__mdset.Models[model_name].section,
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
# end run1


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

    if not __slib.Checker.check_run(
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
            models=__mdset,
            ):
        return None

    model = __mdset.Models[model_name]
    return __stm2.ModelAlgorithm.get_stm_score(
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
# end run2


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

    if not __slib.Checker.check_model_para(
                                    model_type=model_type,
                                    model_ratio=model_ratio_pdf,
                                    model_section=model_section,
                                    model_desc='',
                                    logger=logger,
                                    ):
        return None

    if not __slib.Checker.check_df_cols(
                                df=df,
                                cols=cols,
                                raw_score_range=(raw_score_min, raw_score_max),
                                logger=logger,
                                ):
        return None

    # check strategy
    if not __slib.Checker.check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_sort_order=mode_sort_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            logger=logger,
            models=__mdset
            ):
        return None

    return __stm2.ModelAlgorithm.get_stm_score(
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
