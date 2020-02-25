# coding: utf-8


"""
about module main:
    (1) function: run
    run() is an interface to call stm1 or stm2
    return model(stmlib.ScoreTransform) or result(stm2.get_stm_score.Result: (maptable, df))
    if model_name_in ['zhejiang', 'shanghai', , 'beijing', 'tianjin',
                      'shandong', 'guangdong', 'p7', 'h900', 'hainan',
                      'hn300plt1', 'hn300plt2', 'hn300plt3'
                      ]
        call stm1.PltScore
    then
        call stm2.get_stm_score

    (2) function: runf
    runf(confname) is an interface to call stm1/stm2 with model_name, df, cols
    and other parameters(score range, strategies, out score decimal digits)
    runf need to call a config file to set model parameters, the filename is confname

    (3) function: run1
    run1 is an interface to call stm1 with model name, df, cols and other parameters

    (4) function: run2
    run2 is an interface to call stm2 with model name, df, cols and other parameters

    (5) function: run2p
    run_para() is an interface to call stm2 with df, cols, model_ratio, model_section, model_type,
    and other parameters(score range, strategies, out score decimal digits).
    used to test algorithm commonly.

    (6) function: newconfig
    use newconfig(confname) to create a config file with name=confname

    (7) function: testdata
    create test score data: DataFrame
"""


import time as __time
import os as __os
from collections import namedtuple as __namedtuple
import importlib as __pb
# import numpy as __np

from stm import stmlib as __slib, stm1 as __stm1, stm2 as __stm2, models as __models
__stm_modules = [__slib, __stm1, __stm2, __models]


def run(
        model='shandong',
        df=None,
        cols=None,
        cfg=None,
        mode_score_order='d',
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        logname=None,
        logdisp=True,
        logfile=False,
        loglevel='info',
        logdata=False,
        verify=False,
        value_raw_score_min=0,
        value_raw_score_max=100,
        value_out_score_decimals=0,
        value_tiny_value=10**-10,
        ):

    """
    ---
    基本参数：
        model: str, 模型名称
        df: DataFrame, 原始分数数据
        cols: list of str, 需要转换的原始分数列（字段）, 某考试科目的分数，可以同时指定多个科目，如 ['km1', 'km2']
        cfg: str, 配置文件名称
        注： 如果配置了cfg，则优先使用配置文件的选项进行计算，否则使用其他参数的设置进行计算
             将cfg设置为第4个参数，是为了强制使用cfg=filename形式调用
             模型名称model、分数数据df、分数数据列cols, 必须同时设置，模型名称可以使用缺省值shandong，其它两个参数必须设置
    ---
    算法策略参数：
          mode_score_order: 分数排序方式 score sort method: descending or ascending
                      'd': 降序方式排序，从高分数开始计算
                      'a': 升序方式排序，从低分数开始计算

          mode_ratio_prox: 比例值逼近方式 the mode to proxmate ratio value of raw score points
                         搜索对应比例的确定等级区间分值点的方式
              'upper_min': 小于该比例值的分值中最大的值   get score with min value in bigger percentile
              'lower_max': 大于该比例值的分值中最小的值   get score with max value in less percentile
               'near_min': 最接近该比例值的分值中最小的值 get score with min value in near percentile
               'near_max': 最接近该比例值的分值中最大的值 get score with max value in near percentile

          mode_ratio_cumu: 比例是否累积 use or not cumulative section ratio to locate section point
                    'yes': 以区间比例累计方式搜索 look up ratio with cumulative ratio
                     'no': 以区间比例独立方式搜索 look up ratio with interval ratio respectively

          mode_section_point_first: 第一个端点的取值方式
                            'real': 取实际值，即实际得分的最高分（descending）或最低分数(ascending)
                         'defined': 取定义值，即卷面定义的最高分value_raw_score_min（descending）或最低分数value_raw_score_max(ascending)

          mode_section_point_start: 起始端点的取值方式（第一个区间开始点使用mode_section_point_first确定）
                            'step': 取顺延值，即上一个区间的末端点值加1（descending）或减1(ascending)
                           'share': 取共享值，即上一个区间的末端点值

          mode_section_point_last:  末端点的取值方式
                            'real': 取实际值，即实际得分的最高分（descending）或最低分数(ascending)
                         'defined': 取定义值，即卷面定义的最高分value_raw_score_min（descending）或最低分数value_raw_score_max(ascending)

             mode_section_degraded: 区间退化处理方式
                          'to_max': 取对应输出区间的最大值
                          'to_min': 取对应输出区间的最小值
                         'to_mean': 取对应输出区间的平均值

                 mode_section_lost: 区间消失处理方式
                            'real': 按照实际出现的情况保留消失区间的位置
                             'zip': 将消失区间推压，即移动到最后，使其他区间上移
    ---
    任务控制参数：
        logname: str,  任务名称， 用于日志文件、输出数据文件的前缀
        logdisp: bool, 显示控制，是否显示计算运行结果
        logfile: bool, 日志控制，是否将计算运行结果写入日志文件
        logdata: bool, 数据控制，是否将计算结果写入文件，指转换分数(df_outscore)和转换映射表(df_maptable)
        ---
        loglevel: str, 输出结果的等级：'debug', 'info'，在调试系统时使用
        verify: bool,  是否使用算法验证，即使用两种计算算法对计算结果进行验证， 在调试系统时shiy
    ---
    数值计算参数：
        value_raw_score_min: int,       原始分数的理论最小值
        value_raw_score_max: int,       原始分数的理论最大值
        value_out_score_decimals: int,  输出分数的保留小数位数
        value_tiny_value: float,        微小值，用于过程计算的精度控制， 推荐设为10**-10，最小为10**-14
    ---
    返回值
    返回结果为名称元组：（outdf, maptable, plot, formula）
    outdf: DataFrame,           转换分数结果
    maptable：DataFrame,        转换映射表
    plot: function              转换模型（model）、原始分数分布（raw）、转换分数(out)、分数差异(diff)图示
    formula: dict               转换公式
    ---
    应用示例：
      [1] from stm import main
      [2] m = main.run(model='zhejiang', df=rawscore, col=['ls'])
      [3] m.maptable.head()
      [4] m.outdf.head()
      [5] m.plot('model')
    ---
    """

    # calculation for converting score
    result_namedtuple = __namedtuple('Result', ['ok', 'r1', 'r2'])

    for m in __stm_modules:
        __pb.reload(m)

    mcfg = dict()
    if isinstance(cfg, str):
        mcfg = __read_config(cfg)
        if mcfg:
            model = mcfg['model_name']
            df = mcfg['df']
            cols = mcfg['cols']
            value_raw_score_min = mcfg['value_raw_score_min']
            value_raw_score_max = mcfg['value_raw_score_max']
            mode_ratio_prox = mcfg['mode_ratio_prox']
            mode_ratio_cumu = mcfg['mode_ratio_cumu']
            mode_score_order = mcfg['mode_score_order']
            mode_section_point_first = mcfg['mode_section_point_first']
            mode_section_point_start = mcfg['mode_section_point_start']
            mode_section_point_last = mcfg['mode_section_point_last']
            mode_section_degraded = mcfg['mode_section_degraded']
            mode_section_lost = mcfg['mode_section_lost']
            logname = mcfg['logname']
            logdisp = mcfg['logdisp']
            logfile = mcfg['logfile']
            loglevel = mcfg['loglevel']
            logdata = mcfg['logdata']
            value_out_score_decimals = mcfg['value_out_score_decimals']
            value_tiny_value = mcfg['value_tiny_value']
            verify = mcfg['verify']
        else:
            # print('reading config file fail!')
            return None

    if not isinstance(logname, str):
        task = 'stm'
    else:
        task = logname

    stmlogger = __slib.get_logger(model, logname=task)
    stmlogger.set_level(loglevel)
    stm_no = '  No.' + str(id(stmlogger))
    if logdisp:
        stmlogger.logging_consol = True
    if logfile:
        stmlogger.logging_file = True

    stmlogger.loginfo('\n*** running begin ***')
    stmlogger.loginfo_start('task:' + task + ' model:' + model + stm_no)

    # log--disp--file: config messages form mcfg by reading cfg file
    stmlogger.loginfo('read config in {}'.format(cfg) + '\n' + '-' * 120)
    key_list = list(mcfg.keys())
    for i, k in enumerate(key_list):
        if k == 'df':
            logstr = '   {:25s}: {:10s}'.format(k, str(mcfg[k].columns))
        else:
            logstr = '   {:25s}: {:10s}'.format(k, str(mcfg[k]))
        if i == len(key_list)-1:
            logstr += '\n' + '-' * 120
        stmlogger.loginfo(logstr)

    if not __slib.Checker.check_run(
            model_name=model,
            df=df,
            cols=cols,
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            raw_score_range=(value_raw_score_min, value_raw_score_max),
            out_score_decimal_digits=value_out_score_decimals,
            logger=stmlogger,
            models=__models,
            ):
        stmlogger.loginfo_end('task:' + task + ' model:' + model + stm_no)
        return None
    stmlogger.loginfo('data columns: {}, score fields: {}'.format(list(df.columns), cols))

    model_type = __models.Models[model].type
    # plt, ppt : call stm1.PltScore
    if model_type in ['plt', 'ppt']:
        m1 = __run1(
            name=model,
            df=df,
            cols=cols,
            raw_score_range=(value_raw_score_min, value_raw_score_max),
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            value_out_score_decimals=value_out_score_decimals,
            value_tiny_value=value_tiny_value,
            logger=stmlogger,
            )
        r = result_namedtuple(True, m1, None)
        if verify:
            verify_pass = True
            m2 = __run2(  #
                name=model,
                df=df,
                cols=cols,
                value_raw_score_min=value_raw_score_min,
                value_raw_score_max=value_raw_score_max,
                mode_ratio_prox=mode_ratio_prox,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_score_order=mode_score_order,
                mode_section_point_first=mode_section_point_first,
                mode_section_point_start=mode_section_point_start,
                mode_section_point_last=mode_section_point_last,
                mode_section_degraded=mode_section_degraded,
                mode_section_lost=mode_section_lost,
                value_out_score_decimals=value_out_score_decimals,
                value_tiny_value=value_tiny_value,
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
            r = result_namedtuple(verify_pass, m1, m2)
    # 'pgt': call stmlib.Algorithm.get_stm_score
    else:
        stmlogger.loginfo('run model by stm2, cols={} ... '.format(cols))
        m2 = __run2(   #
                  name=model,
                  df=df,
                  cols=cols,
                  value_raw_score_min=value_raw_score_min,
                  value_raw_score_max=value_raw_score_max,
                  mode_ratio_prox=mode_ratio_prox,
                  mode_ratio_cumu=mode_ratio_cumu,
                  mode_score_order=mode_score_order,
                  mode_section_point_first=mode_section_point_first,
                  mode_section_point_start=mode_section_point_start,
                  mode_section_point_last=mode_section_point_last,
                  mode_section_degraded=mode_section_degraded,
                  value_out_score_decimals=value_out_score_decimals,
                  value_tiny_value=value_tiny_value,
                  logger=stmlogger,
                  )
        r = result_namedtuple(True, None, m2)

    if r.ok:
        t = __time.localtime()
        fno = '_'.join(map(str, [t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec]))
        save_dfscore_name = task + '_df_outscore_' + model + '_' + fno + '.csv'
        save_dfmap_name = task + '_df_maptable_' + model + '_' + fno + '.csv'
        if r.r1 is not None:
            dfscore = r.r1.outdf
            dfmaptable = r.r1.maptable
        else:
            dfscore = r.r2.outdf
            dfmaptable = r.r2.maptable
        if logdata:
            dfscore.to_csv(save_dfscore_name, index=False)
            dfmaptable.to_csv(save_dfmap_name, index=False)
        stmlogger.loginfo('result data: {}\n    score cols: {}'.format(list(dfscore.columns), cols))
        stmlogger.loginfo_end('task:' + task + '  model:{}{} '.format(model, stm_no))
    else:
        stmlogger.loginfo_end('model={} running fail!'.format(model))

    stm_result = __namedtuple('R', ['dfscore', 'maptable', 'plot'])
    if r.ok:
        rr = r.r1 if r.r1 else r.r2
        return stm_result(rr.outdf, rr.maptable, rr.plot)
    return None
# end run


def __run1(
        name='shandong',
        df=None,
        cols=(),
        mode_ratio_prox='upper_min',
        mode_ratio_cumu='no',
        mode_score_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        raw_score_range=(0, 100),
        value_out_score_decimals=0,
        value_tiny_value=10 ** -12,
        logger=None,
        debug=False,
        ):

    ratio_tuple = tuple(x * 0.01 for x in __models.Models[name].ratio)
    model_type = __models.Models[name].type
    m = __stm1.PltScore(name, model_type)
    m.set_data(df=df, cols=cols)
    m.set_para(
        raw_score_ratio=ratio_tuple,
        out_score_section=__models.Models[name].section,
        raw_score_defined_max=max(raw_score_range),
        raw_score_defined_min=min(raw_score_range),
        mode_ratio_prox=mode_ratio_prox,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_score_order=mode_score_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        value_out_score_decimals=value_out_score_decimals,
        value_tiny_value=value_tiny_value,
        logger=logger,
        )
    m.run()
    if debug:
        return m
    else:
        result = __namedtuple('Result', ['outdf', 'maptable', 'plot', 'formula'])
        return result(m.outdf, m.maptable, m.plot, m.result_dict)
# end run1


# get stm score by calling stmlib2.ModelAlgorithm
def __run2(
        name='shandong',
        df=None,
        cols=(),
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_score_order='d',
        mode_section_point_first='real',
        mode_section_point_start='step',
        mode_section_point_last='real',
        mode_section_degraded='to_max',
        mode_section_lost='real',
        value_raw_score_min=0,
        value_raw_score_max=100,
        raw_score_step=1,
        value_out_score_decimals=0,
        value_tiny_value=10**-10,
        logger=None,
        ):
    """
    to calculate out score by calling stmlib2.Algorithm.get_stm_score
    model_name in models_in.Models or models_ext.Models
    """

    if not __slib.Checker.check_run(
            model_name=name,
            df=df,
            cols=cols,
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            raw_score_range=(value_raw_score_min, value_raw_score_max),
            out_score_decimal_digits=value_out_score_decimals,
            logger=logger,
            models=__models,
            ):
        return None

    model = __models.Models[name]
    return __stm2.stm2(     # ModelAlgorithm.get_stm_score
        df=df,
        cols=cols,
        model_ratio_pdf=model.ratio,
        model_section=model.section,
        model_type=model.type,
        value_raw_score_max=value_raw_score_max,
        value_raw_score_min=value_raw_score_min,
        raw_score_step=raw_score_step,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_ratio_prox=mode_ratio_prox,
        mode_score_order=mode_score_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        value_out_score_decimals=value_out_score_decimals,
        value_tiny_value=value_tiny_value,
        logger=logger,
        )
# end run2


# calc stm score by calling methods in stmlib2.ModelAlgorithm
def __run2p(
        df,
        cols,
        model_type=None,
        model_ratio_pdf=None,
        model_section=None,
        value_raw_score_max=100,
        value_raw_score_min=0,
        raw_score_step=1,
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        mode_score_order='d',
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
    :param value_raw_score_max:
    :param value_raw_score_min:
    :param raw_score_step:
    :param mode_ratio_cumu:
    :param mode_ratio_prox:
    :param mode_score_order:
    :param mode_section_point_first:
    :param mode_section_point_start:
    :param mode_section_point_last:
    :param mode_section_degraded:
    :param mode_section_lost:
    :param out_score_decimal_digits:
    :param logger: set logger
    :return: result(outdf, maptable),     out_score_field[col+'_ts'] for col in cols in outdf
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
                                raw_score_range=(value_raw_score_min, value_raw_score_max),
                                logger=logger,
                                ):
        return None

    # check strategy
    if not __slib.Checker.check_strategy(
            mode_ratio_prox=mode_ratio_prox,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order,
            mode_section_point_first=mode_section_point_first,
            mode_section_point_start=mode_section_point_start,
            mode_section_point_last=mode_section_point_last,
            mode_section_degraded=mode_section_degraded,
            mode_section_lost=mode_section_lost,
            logger=logger,
            models=__models
            ):
        return None

    return __stm2.ModelAlgorithm.get_stm_score(
        df=df,
        cols=cols,
        model_ratio_pdf=model_ratio_pdf,
        model_section=model_section,
        model_type=model_type,
        value_raw_score_max=value_raw_score_max,
        value_raw_score_min=value_raw_score_min,
        raw_score_step=raw_score_step,
        mode_ratio_cumu=mode_ratio_cumu,
        mode_ratio_prox=mode_ratio_prox,
        mode_score_order=mode_score_order,
        mode_section_point_first=mode_section_point_first,
        mode_section_point_start=mode_section_point_start,
        mode_section_point_last=mode_section_point_last,
        mode_section_degraded=mode_section_degraded,
        mode_section_lost=mode_section_lost,
        value_out_score_decimals=out_score_decimal_digits,
        logger=logger,
        )
# end--run2p


def models(name=None):
    Models = __models.Models
    if isinstance(name, str):
        if name in Models:
            v = Models[name]
            print('{:<15s} {},  {} '.format(name, v.type, v.desc))
            print('{:<15s} {}'.format(' ', v.ratio))
            print('{:<15s} {}'.format('', v.section))
        else:
            print('name={} not found!'.format(name))
        return
    elif name is not None:
        print('invalid name={}'.format(name))
        return

    for k in Models:
        v = Models[k]
        print('{:<15s} {},  {} '.format(k, v.type, v.desc))
        print('{:<15s} {}'.format(' ', v.ratio))
        print('{:<15s} {}'.format('', v.section))


def make_config_file(filename='stm.cfg'):
    if __slib.isfilename(filename):
        __slib.make_config_file(filename)
        return True
    else:
        print('invalid file name!')
        return False

def __read_config(filename='stm.cfg'):

    if isinstance(filename, str):
        if '.' not in filename:
            filename += '.cfg'
    else:
        print('Invalid filename={}'.format(filename))
        return False

    if not __os.path.isfile(filename):
        print('conf file: {} not found!'.format(filename))
        return False

    mcfg = __slib.read_config_file(filename)
    if not mcfg:
        print('read config file {} fail!'.format(filename))
        return False

    # use new model when no model defined, that is in modelset
    if 'model_new_set' in mcfg:
        # new model in config-file
        if mcfg['model_new_check']:
            mcfg.update({'model_name': mcfg['model_new_name']})
            __models.Models.update(
                {mcfg['model_name']: __models.ModelFields(
                    mcfg['model_new_type'],
                    mcfg['model_new_ratio'],
                    mcfg['model_new_section'],
                    mcfg['model_new_desc']
                )}
            )
            print('new model [{}] check-in!'.format(mcfg['model_new_name']))
        # no model can be used
        else:
            print('bad new model definition in config file !')
            return False

    if mcfg['df'] is None:
        print('reading data file to df fail!')
        return False
    if mcfg['cols'] is None:
        print('no data columns assigned!')
        return False

    return mcfg
