# -*- utf-8 -*-


# comments to stm
"""
    2018.09.24 -- 2018.11
    2019.09.03 --
    designed for new High Test grade score model
    also for shandong interval linear transform

    stm module description stm模块说明：

    [functions] 模块中的函数
       run(name, df, col, ratio_list, grade_max, grade_diff, raw_score_max, raw_score_min,
           out_score_decimal=0, mode_ratio_seek='near', mode_ratio_cumu='yes')
          运行各个模型的调用函数 calling model function
          ---
          参数描述
          name:= 'shandong'/'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tai'
          调用山东、上海、浙江、北京、天津、广州、海南、...等模型进行分数转换
          caculate shandong... model by name = 'shandong' / 'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tai'
          -
          name:= 'zscore'/'t_score'/'tlinear'
          计算Z分数、T分数、线性转换T分数
          caculate Z,T,liear T score by name = 'zscore'/ 't_score' / 'tlinear'
          --
          df: input raw score data, type DataFrame of pandas
          输入原始分数数据，类型为DataFrame
          --
          col: score field to calculate in df
          计算转换分数的字段表
          --
          ratio_list: ratio list including percent value for each interval of grade score
          划分等级的比例表
          --
          grade_max: max value of grade score
          最大等级分数
          --
          grade_diff: differentiao value of grade score
          等级分差值
          --
          raw_score_max: raw score max value
          最大原始分数
          --
          raw_score_min: raw score min value
          最小原始分数
          --
          out_score_decimal: grade score precision, decimal digit number
          输出分数精度，小数位数
          --
          mode_ratio_seek: how to approxmate score points of raw score for each ratio vlaue
          计算等级时的逼近方式（目前设计的比例值逼近策略)：
              'upper_min': get score with min value in bigger 小于该比例值的分值中最大的值
              'lower_max': get score with max value in less 大于该比例值的分值中最小的值
              'near':   get score with nearest ratio 最接近该比例值的分值（分值）
              'near_min': get score with min value in near 最接近该比例值的分值中最小的值
              'near_max': get score with max value in near 最接近该比例值的分值中最大的值
              注1：针对等级划分区间，也可以考虑使用ROUND_HALF_UP，即靠近最近，等距时向上靠近
              注2：搜索顺序分为Big2Small和Small2Big两类，区间位精确的定点小数，只有重合点需要策略（UP或DOWN）

              拟改进为（2019.09.09） mode_ratio_seek：
              'near':    look up the nearest in all ratios to given-ratio 最接近的比例
              'upper_min':  look up the maximun in ratios which is less than given-ratio 小于给定比例的最大值
              'lower_max':  look up the minimun in ratios which is more than given-ratio 大于给定比例的最小值

              仍然使用四种模式(2019.09.25)： upper_min, lower_max, near_min, near_max

          拟增加比例累加控制(2019.09.09)：
          mode_ratio_cumu:
              'yes': 以区间比例累计方式搜索 look up ratio with cumulative ratio
              'no':  以区间比例独立方式搜索 look up ratio with interval ratio individually

          ---
          usage:调用方式
          [1] import pyex_stm as stm
          [2] m = stm.run(name='shandong', df=data, col=['ls'])
          [3] m.report()
          [4] m.output.head()
          [5] m.save_out_data_to_csv

       plot()
          山东、浙江、上海、北京、天津、广东、湖南方案等级转换分数分布直方图
          plot models distribution hist graph including shandong,zhejiang,shanghai,beijing,tianjin

       round45i(v: float, dec=0)
          四舍五入函数
          function for rounding strictly at some decimal position
          v 输入浮点数， dec：保留小数位数，缺省为0

       get_norm_dist_table(size=400, std=1, mean=0, stdnum=4)
          生成具有指定记录数（size=400）、标准差(std=1)、均值(mean=0)、截止标准差数（最小最大）(stdnum=4)的正态分布表
          create norm data dataframe with assigned scale, mean, standard deviation, std range

       get_norm_dist_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000, decimal=6)
          生成具有指定均值(mean=70)、标准差(std=10)、最大值(maxvalue=100)、最小值(minvalue=0)、
          样本数(size=1000)、保留小数位（decimal=6）的数据样本集(pandas.DataFrame)
          create sample data set according to assigned mean, std, maxvalue, minvalue, size, decimal

    [classes] 模块中的类
       PltScore: 分段线性转换模型, 山东省新高考改革使用 shandong model
       GradeScore: 等级分数转换模型, 浙江、上海、天津、北京使用 zhejiang shanghai tianjin beijing model
       TaiScore: 台湾等级分数模型 Taiwan college entrance test and middle school achievement test model
       Zscore: Z分数转换模型 zscore model
       Tscore: T分数转换模型 t_score model
       Tlinear: T分数线性转换模型 t_score model by linear transform mode
       SegTable: 计算分段表模型 segment table model

    [CONSTANTS] 模块中的常量
        各省市等级分数转换比例设置，山东省区间划分设置
        CONST_ZHEJIANG_RATIO = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        CONST_SHANGHAI_RATIO = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
        CONST_BEIJING_RATIO = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
        CONST_TIANJIN_RATIO = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]
        CONST_SHANDONG_RATIO = [3, 7, 16, 24, 24, 16, 7, 3]
        CONST_SHANDONG_SEGMENT = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
        CONST_SS7_RATIO = [2, 13, 35, 35, 15]
        CONST_SS7_SEGMENT = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]
"""


# built-in import
import copy
import time
import os
import warnings
import fractions as fr
from collections import namedtuple
import bisect as bst
import array
# import functools as func
# import decimal as dc


# external import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
import seaborn as sbn
from pyex_tool import pyex_ptt as ptt


warnings.filterwarnings('ignore')


# some constants for models: score grade ratio, shandong grade score interval
CONST_ZHEJIANG_RATIO = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
CONST_ZHEJIANG_SEGMENT = [(100-i*3, 100-i*3) for i in range(21)]
CONST_SHANGHAI_RATIO = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
CONST_SHANGHAI_SEGMENT = [(70-i*3, 70-i*3) for i in range(11)]
CONST_BEIJING_RATIO = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
CONST_BEIJING_SEGMENT = [(100-i*3, 100-i*3) for i in range(21)]
CONST_TIANJIN_RATIO = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]
CONST_TIANJIN_SEGMENT = [(100-i*3, 100-i*3) for i in range(21)]

# ShanDong
# 8 levels, [3%, 7%, 16%, 24%, 24%, 16%, 7%, 3%]
# 8 segments: [100,91], ..., [30,21]
CONST_SHANDONG_RATIO = [3, 7, 16, 24, 24, 16, 7, 3]
CONST_SHANDONG_SEGMENT = [(100-i*10, 100-i*10-9) for i in range(8)]

# GuangDong
#   predict: mean = 70.21, std = 20.95
CONST_GUANGDONG_RATIO = [17, 33, 33, 15, 2]
CONST_GUANGDONG_SEGMENT = [(100, 83), (82, 71), (70, 59), (58, 41), (40, 30)]

# JIANGSU, FUJIAN, HUNAN, HUBEI, CHONGQING, HEBEI, LIAONING
#   ration=(15%、35%、35%、13%, 2%), 5 levels
#   segment=(30～40、41～55、56～70、71～85、86～100)
#   predict: mean = 70.24, std = 21.76
#            mean = sum([x/100*sum(y)/2 for x,y in zip(SS7ratio,SS7segment)])
#            std = math.sqrt(sum([(sum(y)/2-mean)**2 for x,y in zip(SS7ratio,SS7segment)])/5)
CONST_SS7_RATIO = [15, 35, 35, 13, 2]
CONST_SS7_SEGMENT = [(100, 86), (85, 71), (70, 56), (55, 41), (40, 30)]

# Hainan standard score 100-900
norm_cdf = [sts.norm.cdf((v-500)/100) for v in range(100, 901)]
CONST_HAINAN_RATIO = [(norm_cdf[i] - norm_cdf[i-1])*100 if i > 0    # set start ratio to (1 - cdf(-4))*100
                      else norm_cdf[i]*100 for i in range(801)]
CONST_HAINAN_RATIO[-1] = 100 - sum(CONST_HAINAN_RATIO[:-1])         # ensure to sum==100
CONST_HAINAN_SEGMENT = [(s, s) for s in range(900, 99, -1)]

# Hainan2 standard score for new Gaokao 60-300
_start_point, end_point = 60, 300
_mean = (_start_point + end_point) / 2
_std = (_mean - _start_point) / 4
norm_cdf = [sts.norm.cdf((v-_mean)/_std) for v in range(_start_point, end_point + 1)]
CONST_HAINAN_RATIO2 = [(norm_cdf[i] - norm_cdf[i-1])*100 if i > 0    # set start ratio to (1 - cdf(-4))*100
                      else norm_cdf[i]*100 for i in range(end_point - _start_point + 1)]
CONST_HAINAN_RATIO2[-1] = 100 - sum(CONST_HAINAN_RATIO2[:-1])         # ensure to sum==100
CONST_HAINAN_SEGMENT2 = [(s, s) for s in range(end_point, _start_point - 1, -1)]

# Hainan3 using plt for 60-300
# f = sts.norm.cdf
# CONST_HAINAN_RATIO3 = [(f(i+1)-f(i) if i < 3 else 1-f(i)) if i > -4 else f(i+1) for i in range(-4, 4)]
CONST_HAINAN_RATIO3 = [0.14, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.14]
CONST_HAINAN_SEGMENT3 = [(x, x-30+1 if x > 90 else x-30) for x in range(300, 60, -30)]


PltRatioSeg_namedtuple = namedtuple('Plt', ['ratio', 'seg'])
plt_models_dict = {
    'zhejiang': PltRatioSeg_namedtuple(CONST_ZHEJIANG_RATIO, CONST_ZHEJIANG_SEGMENT),
    'shanghai': PltRatioSeg_namedtuple(CONST_SHANGHAI_RATIO, CONST_SHANGHAI_SEGMENT),
    'beijing': PltRatioSeg_namedtuple(CONST_BEIJING_RATIO, CONST_BEIJING_SEGMENT),
    'tianjin': PltRatioSeg_namedtuple(CONST_TIANJIN_RATIO, CONST_TIANJIN_SEGMENT),
    'shandong': PltRatioSeg_namedtuple(CONST_SHANDONG_RATIO, CONST_SHANDONG_SEGMENT),
    'guangdong': PltRatioSeg_namedtuple(CONST_GUANGDONG_RATIO, CONST_GUANGDONG_SEGMENT),
    'ss7': PltRatioSeg_namedtuple(CONST_SS7_RATIO, CONST_SS7_SEGMENT),
    'hainan': PltRatioSeg_namedtuple(CONST_HAINAN_RATIO, CONST_HAINAN_SEGMENT),
    'hainan2': PltRatioSeg_namedtuple(CONST_HAINAN_RATIO2, CONST_HAINAN_SEGMENT2),
    'hainan3': PltRatioSeg_namedtuple(CONST_HAINAN_RATIO3, CONST_HAINAN_SEGMENT3)
    }
stm_strategies_dict = {
    'mode_score_order': ['ascending', 'descending'],
    'mode_ratio_seek': ['upper_min', 'lower_max', 'near_max', 'near_min'],
    'mode_ratio_cumu': ['yes', 'no'],
    'mode_seg_degraded': ['max', 'min', 'mean'],
    'mode_score_max': ['map_to_max', 'map_by_ratio'],
    'mode_score_min': [ 'map_to_min', 'map_by_ratio'],
    'mode_score_zero': ['map_to_min', 'map_by_ratio', 'ignore'],
    'mode_score_empty': ['map_to_min', 'map_to_max', 'map_to_mean', 'ignore'],
    'mode_endpoint_share': ['yes', 'no']
    }
stm_models_name = list(plt_models_dict.keys()) + ['zscore', 'tscore', 'tai', 'tlinear']


def about():
    print(__doc__)


# interface to use model for some typical application
def run(
        name='shandong',
        data=None,
        cols=(),
        mode_ratio_seek='upper_min',
        mode_ratio_cumu='no',
        mode_score_order='descending',
        raw_score_range=(0, 100),
        out_score_decimal=0
        ):
    """
    :param name: str, model name, values: 'shandong', 'shanghai', 'shandong', 'beijing', 'tianjin', 'zscore', 'tscore',
                                          'tlinear', 'guangdong', 'SS7', 'hainan', 'hainan2'
                 default = 'shandong'
    :param data: dataframe, raw score data, score field type must be int or float
                 default = None
    :param cols: list, which elements are score fields in data,
                 default = None
    :param raw_score_range: tuple, raw score value range,
                            default = (0, 100)
    :param out_score_decimal: int, decimal digits of output score
                              default = 0, thar means out score type is int
    :param mode_ratio_seek: string, strategy to locate ratio, values: 'lower_max', 'upper_min', 'near_max', 'near_min'
                           default='upper_min'
    :param mode_ratio_cumu: string, strategy to cumulate ratio, values:'yes', 'no'
                           default='no'
    :param mode_score_order: string, strategy to sort score, values: 'descending', 'ascending'
                             default='descending'
    :return: model
    """
    # check name
    name = name.lower()
    if name.lower() not in stm_models_name:
        print('invalid name, not in {}'.format(stm_models_name))
        return

    # check input data
    if type(data) != pd.DataFrame:
        if type(data) == pd.Series:
            raw_data = pd.DataFrame(data)
        else:
            print('no score dataframe!')
            return
    else:
        raw_data = data

    # check col
    if isinstance(cols, str):
        cols = cols,
    elif type(cols) not in (list, tuple):
        print('invalid cols type:{}!'.format(type(cols)))
        return

    # check mode_ratio_seek
    if mode_ratio_seek not in ['lower_max', 'upper_min', 'near_min', 'near_max']:
        print('invalid approx mode: {}'.format(mode_ratio_seek))
        print('  valid approx mode: lower_max, upper_min, near_min, near_max')
        return
    if mode_ratio_cumu not in ['yes', 'no']:
        print('invalid cumu mode(yes/no): {}'.format(mode_ratio_cumu))
        return

    # plt score models
    if name in plt_models_dict.keys():
        ratio_list = [x*0.01 for x in plt_models_dict[name].ratio]
        plt_model = PltScore()
        plt_model.model_name = name
        plt_model.out_decimal_digits = 0
        plt_model.set_data(raw_data=raw_data, cols=cols)
        plt_model.set_para(
            raw_score_ratio_list=ratio_list,
            out_score_seg_list=plt_models_dict[name].seg,
            raw_score_range=raw_score_range,
            mode_ratio_seek=mode_ratio_seek,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order,
            out_decimal_digits=out_score_decimal
            )
        plt_model.run()
        return plt_model

    if name in ('tai', 'taiwan'):
        m = GradeScoreTai()
        m.grade_num = 15    # TaiWan use 15 levels grade score system
        m.set_data(raw_data=raw_data,
                   cols=cols)
        m.set_para(raw_score_max=raw_score_range[1],
                   raw_score_min=raw_score_range[0])
        m.run()
        return m

    if name in ('zscore', 'Z', 'z'):
        zm = Zscore()
        zm.model_name = name
        zm.set_data(raw_data=raw_data, cols=cols)
        zm.set_para(std_num=4, raw_score_range=raw_score_range)
        zm.run()
        zm.report()
        return zm

    if name in ('tscore', 'T', 't'):
        tm = Tscore()
        tm.model_name = name
        tm.set_data(raw_data=raw_data, cols=cols)
        tm.set_para(raw_score_range=raw_score_range)
        tm.run()
        tm.report()
        return tm

    if name == 'tlinear':
        tm = TscoreLinear()
        tm.model_name = name
        tm.set_data(raw_data=raw_data, cols=cols)
        tm.set_para(raw_score_max=raw_score_range[1],
                    raw_score_min=raw_score_range[0])
        tm.run()
        tm.report()
        return tm


def plot_stm(font_size=12, hainan='900'):
    _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'ss7', 'hainan']
    if hainan == '300':
        _names.remove('hainan')
        _names.append('hainan2')
    ms_dict = dict()
    for _name in _names:
        ms_dict.update({_name: get_stm_model_describe(name=_name)})

    plot.figure('New Gaokao Score Models: name(mean, std, skewness)')
    plot.rcParams.update({'font.size': font_size})
    for i, k in enumerate(_names):
        plot.subplot(240+i+1)
        _wid = 2
        if k in ['shanghai']:
            x_data = range(40, 71, 3)
        elif k in ['zhejiang', 'beijing', 'tianjin']:
            x_data = range(40, 101, 3)
        elif k in ['shandong']:
            x_data = [x for x in range(26, 100, 10)]
            _wid = 8
        elif k in ['guangdong']:
            x_data = [np.mean(x) for x in plt_models_dict[k].seg][::-1]
            _wid = 10
        elif k in ['ss7']:
            x_data = [int(np.mean(x)) for x in plt_models_dict[k].seg][::-1]
            _wid = 10
        elif k in ['hainan']:
            x_data = [x for x in range(100, 901)]
            _wid = 1
        elif k in ['hainan2']:
            x_data = [x for x in range(60, 301)]
            _wid = 1
        else:
            raise ValueError(k)
        plot.bar(x_data, plt_models_dict[k].ratio[::-1], width=_wid)
        plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))


def get_stm_model_describe(name='shandong'):
    __ratio = plt_models_dict[name].ratio
    __seg = plt_models_dict[name].seg
    if name == 'hainan':
        __mean, __std, __skewness = 500, 100, 0
    elif name == 'hainan2':
        __mean, __std, __skewness = 180, 30, 0
    else:
        samples = []
        [samples.extend([np.mean(s)]*int(__ratio[i])) for i, s in enumerate(__seg)]
        __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))

    # __mean = sum([r / 100 * np.mean(s) for r, s in zip(__ratio, __seg)])
    # __std = np.sqrt(sum([__ratio[i] * (np.mean(s) - __mean) ** 2
    #                     for i, s in enumerate(__seg)]) / 100)
    # __skew_numerator = sum([__ratio[i] * (np.mean(s) - __mean)**3
    #                        for i, s in enumerate(__seg)])
    # __skewness = __skew_numerator / sum([__ratio[i] * (np.mean(s)-__mean)**2
    #                                      for i, s in enumerate(__seg)])**(3/2)
    # if __skewness < 1e-4:
    #     __skewness = 0

    return __mean, __std, __skewness


# Score Transform Model Interface
# Abstract class
class ScoreTransformModel(object):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    基于该类的子类（转换分数模型）：
        Z分数非线性模型(Zscore)
        T分数非线性模型(Tscore）
        T分数线性模型（TscoreLinear),
        等级分数模型(GradeScore)
        山东省新高考转换分数模型（PltScore）（分段线性转换分数）
        param model_name, type==str
        param raw_data: raw score data, type==datafrmae
        param col: fields in raw_data, assign somr subjects score to transform
        param out_data: transform score data, type==dataframe
    """
    def __init__(self, model_name=''):
        self.model_name = model_name

        self.raw_data = pd.DataFrame()
        self.cols = []
        self.raw_score_range = (0, 100)
        self.raw_score_min = None
        self.raw_score_max = None

        self.out_data = pd.DataFrame()
        self.out_decimal_digits = 0
        self.out_report_doc = ''
        self.map_table = pd.DataFrame()

        self.sys_pricision_decimals = 8

    def set_data(self, raw_data=None, cols=None):
        raise NotImplementedError()

    def set_para(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if not isinstance(self.raw_data, pd.DataFrame):
            print('rawdf is not dataframe!')
            return False
        if (type(self.cols) != list) | (len(self.cols) == 0):
            print('no score fields assigned!')
            return False
        for sf in self.cols:
            if sf not in self.raw_data.columns:
                print('error score field {} !'.format(sf))
                return False
        return True

    def check_parameter(self):
        return True

    def run(self):
        if not self.check_data():
            print('check data find error!')
            return False
        if not self.check_parameter():
            print('check parameter find error!')
            return False
        return True

    def read_raw_data_from_csv(self, filename=''):
        if not os.path.isfile(filename):
            print('{} not valid file name'.format(filename))
            return
        self.raw_data = pd.read_csv(filename)

    def save_out_data_to_csv(self, filename):
        self.out_data.to_csv(filename, index=False)

    def save_report_to_file(self, filename):
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(self.out_report_doc)

    def save_map_table_to_file(self, filename):
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(ptt.make_mpage(self.map_table, page_line_num=50))

    def report(self):
        raise NotImplementedError()

    def plot(self, mode='raw'):
        # implemented plot_out, plot_raw score figure
        if mode.lower() == 'out':
            self.__plot_out_score()
        elif mode.lower() == 'raw':
            self.__plot_raw_score()
        else:
            # print('error mode={}, valid mode: out or raw'.format(mode))
            return False
        return True

    def __plot_out_score(self):
        if not self.cols:
            print('no field:{0} assign in {1}!'.format(self.cols, self.raw_data))
            return
        # plot.figure(self.model_name + ' out score figure')
        labelstr = 'Output Score '
        for fs in self.cols:
            plot.figure(fs)
            if fs + '_plt' in self.out_data.columns:  # find sf_out_score field
                sbn.distplot(self.out_data[fs + '_plt'])
                plot.title(labelstr + fs)
            elif fs + '_grade' in self.out_data.columns:  # find sf_out_score field
                sbn.distplot(self.out_data[fs + '_grade'])
                plot.title(labelstr + fs)
            else:
                print('mode=out only for plt and grade model!')
        return

    def __plot_raw_score(self):
        if not self.cols:
            print('no field assign in rawdf!')
            return
        labelstr = 'Raw Score '
        for sf in self.cols:
            plot.figure(sf)
            sbn.distplot(self.raw_data[sf])
            plot.title(labelstr + sf)
        return


# piecewise linear transform model
    # ShanDong Model Analysis
    # ratio = [3, 7, 16, 24, 24, 16, 7, 3], grade = [(21, 30),(31, 40), ..., (91, 100)]
    # following is estimation to std:
        # according to percent
        #   test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #   test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #   test std=15.950713502      at 30    Zcdf(-30/std)=0.03
        # according to std
        #   cdf(100)= 0.99496 as std=15.54375, 0.9939 as std=15.9507
        #   cdf(90) = 0.970(9)79656 as std=15.9507135,  0.972718 as std=15.606,  0.9731988 as std=15.54375
        #   cdf(80) = 0.900001195 as std=15.606,  0.9008989 as std=15.54375
        #   cdf(70) = 0.0.73999999697 as std=15.54375
        #   cdf(60) = 0.0
        #   cdf(50) = 0.26  +3.027*E-9 as std=15.54375
        #   cdf(40) = 0.0991 as std=15.54375
        #   cdf(30) = 0.0268 as std=15.54375
        #   cdf(20) = 0.0050 as std=15.54375
        # some problems:
        #   p1: std scope in 15.5-16
        #   p2: cut percent at 20, 100 is a little big, so std is reduced
        #   p3: percent at 30,40 is a bit larger than normal according to std=15.54375
        # on the whole, fitting is approximate fine
        # set model score percentages and endpoints
        # get approximate normal distribution
        # according to percent , test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #                        test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #                        test std=15.950713502      at 30    Zcdf(-30/std)=0.03
        # according to std
        #   cdf(100)= 0.99496           as std=15.54375,    0.9948      as std=15.606       0.9939    as std=15.9507
        #   cdf(90) = 0.970(9)79656     as std=15.9507135   0.97000     as std=15.54375     0.972718    as std=15.606
        #   cdf(80) = 0.900001195       as std=15.606,      0.9008989   as std=15.54375
        #   cdf(70) = 0.0.73999999697   as std=15.54375
        #   cdf(60) = 0.0
        #   cdf(50) = 0.26+3.027*E-9    as std=15.54375
        #   cdf(40) = 0.0991            as std=15.54375
        #   cdf(30) = 0.0268            as std=15.54375
        #   cdf(20) = 0.0050            as std=15.54375
        # ---------------------------------------------------------------------------------------------------------
        #     percent       0      0.03       0.10      0.26      0.50    0.74       0.90      0.97       1.00
        #   std/points      20      30         40        50        60      70         80        90         100
        #   15.54375    0.0050   0.0268       0.0991   [0.26000]   0    0.739(6)  0.9008989  0.97000    0.99496
        #   15.6060     0.0052   0.0273      [0.09999]  0.26083    0    0.73917   0.9000012  0.97272    0.99481
        #   15.9507     0.0061  [0.0299(5)]   0.10495   0.26535    0    0.73465   0.8950418  0.970(4)   0.99392
        # ---------------------------------------------------------------------------------------------------------
        # on the whole, fitting is approximate fine
        # p1: std scope in 15.54 - 15.95
        # p2: cut percent at 20, 100 is a little big, std would be reduced
        # p3: percent at 30 is a bit larger than normal according to std=15.54375, same at 40
        # p4: max frequency at 60 estimation:
        #     percentage in 50-60: pg60 = [norm.pdf(0)=0.398942]/[add:pdf(50-60)=4.091] = 0.097517
        #     percentage in all  : pga = pg60*0.24 = 0.023404
        #     peak frequency estimation: 0.0234 * total_number
        #          max number at mean nerghbor point:200,000-->4680,   300,000 --> 7020
    # GuangDong Model Analysis
    # ratio = [2, 15, 33, 33, 17]
    # gradescore = [(30, 40), (41, 58), (59, 70), (71, 82), (83, 100)]
    # meanscore ~= 35*0.02+48.5*0.13+64.5*0.35+76.5*0.35+92.5*0.15 = 70.23
    # SS7 Model Analysis
    # ratio = [2, 13, 35, 35, 15]
    # gradescore = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]
    # meanscore ~= 35*0.02+48*0.13+63*0.35+78*0.35+93*0.15 = 70.24

class PltScore(ScoreTransformModel):
    """
    PltModel:
    linear transform from raw-score to grade-score at each intervals divided by preset ratios
    set ratio and intervals according to norm distribution property
    get a near normal distribution
    """

    def __init__(self):
        # intit raw_data, out_data, model_name
        super(PltScore, self).__init__('plt')

        # new properties for shandong model
        self.raw_score_ratio_cum = []
        self.out_score_points = []
        self.out_decimal_digits = 0
        self.out_score_max = None
        self.out_score_min = None

        # para
        self.strategy_dict = {
            'mode_ratio_seek': 'upper_min',
            'mode_ratio_cumu': 'yes',
            'mode_score_order': 'descending',
            'mode_seg_degraded': 'max',
            'mode_score_zero': 'map_by_ratio',
            'mode_score_max': 'map_to_max',
            'mode_score_min': 'map_by_ratio',
            'mode_score_empty': 'ignore',
            'mode_endpoint_share': 'no'
        }

        # result
        self.seg_model = None
        self.map_table = pd.DataFrame()
        self.result_raw_endpoints = []
        self.result_ratio_dict = {}
        self.result_formula_coeff = {}
        self.result_formula_text_list = ''
        self.result_dict = {}

    # plt
    def set_data(self, raw_data=None, cols=None):

        # check and set rawdf
        if type(raw_data) == pd.Series:
            self.raw_data = pd.DataFrame(raw_data)
        elif type(raw_data) == pd.DataFrame:
            self.raw_data = raw_data
        else:
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set out_data
        if not cols:
            self.cols = [s for s in raw_data]
        elif type(cols) != list:
            print('col set fail!\n not a list!')
            return
        elif sum([1 if sf in raw_data else 0 for sf in cols]) != len(cols):
            print('col set fail!\n field must in rawdf.columns!')
            return
        else:
            self.cols = cols

    # plt
    def set_para(self,
                 raw_score_ratio_list=None,
                 out_score_seg_list=None,
                 raw_score_range=(0, 100),
                 mode_ratio_seek='upper_min',
                 mode_ratio_cumu='yes',
                 mode_score_order='descending',
                 mode_endpoint_share='no',
                 out_decimal_digits=None):
        if (type(raw_score_ratio_list) not in
           (list, tuple)) | (type(out_score_seg_list) != list):
            print('input score points or output score points is not list!')
            return
        if len(raw_score_ratio_list) != len(out_score_seg_list):
            print('the number of input score points is not same as output score points!')
            return
        if mode_ratio_cumu not in 'yes, no':
            print('mode_ratio_cumu value error:{}'.format(mode_ratio_cumu))

        if isinstance(out_decimal_digits, int):
            self.out_decimal_digits = out_decimal_digits

        if mode_score_order in ['descending', 'd']:
            raw_p = raw_score_ratio_list
            out_pt = out_score_seg_list
        else:
            raw_p = raw_score_ratio_list[::-1]
            out_pt = out_score_seg_list[::-1]
        self.out_score_points = [x[::-1] for x in out_pt]
        self.out_score_points = out_score_seg_list
        self.raw_score_ratio_cum = [sum(raw_p[0:x + 1]) for x in range(len(raw_p))]

        self.strategy_dict['mode_ratio_seek'] = mode_ratio_seek
        self.strategy_dict['mode_ratio_cumu'] = mode_ratio_cumu
        self.strategy_dict['mode_score_order'] = mode_score_order
        self.strategy_dict['mode_endpoint_share'] = mode_endpoint_share

    def check_parameter(self):
        if not self.cols:
            print('no score field assign in col!')
            return False
        if (type(self.raw_score_ratio_cum) != list) | (type(self.out_score_points) != list):
            print('raw_scorepoints or stdscorepoints is not list type!')
            return False
        if (len(self.raw_score_ratio_cum) != len(self.out_score_points)) | \
                len(self.raw_score_ratio_cum) == 0:
            print('len is 0 or not same for raw score percent and std score points list!')
            return False
        return True
    # --------------data and para setting end

    # plt score run
    def run(self):

        print('stm-run begin...\n'+'='*100)
        stime = time.time()

        # check valid
        if not super(PltScore, self).run():
            return

        # if self.raw_score_max is None:
        #     self.raw_score_max = max(self.raw_data[self.cols].max())
        # if self.raw_score_min is None:
        #     self.raw_score_min = min(self.raw_data[self.cols].min())

        if self.out_score_points is not None:
            self.out_score_max = max([max(x) for x in self.out_score_points])
            self.out_score_min = min([min(x) for x in self.out_score_points])

        # calculate seg table
        print('--- calculating map_table ...')
        _segsort = self.strategy_dict['mode_score_order']
        self.seg_model = run_seg(
                  raw_data=self.raw_data,
                  cols=self.cols,
                  segmax=self.raw_score_range[1],
                  segmin=self.raw_score_range[0],
                  segsort='a' if _segsort in ['ascending', 'a'] else 'd',
                  segstep=1,
                  display=False,
                  usealldata=False
                  )
        self.map_table = self.seg_model.out_data   # .copy(deep=True)

        # create field_fr in map_table
        #   strange error!!: some seg percent to zero
        #   self.map_table[f+'_percent'] = self.map_table[f+'_fr'].apply(lambda x: float(x))
        for f in self.cols:
            max_sum = max(self.map_table[f+'_sum'])
            max_sum = 1 if max_sum == 0 else max_sum
            self.map_table[f+'_fr'] = \
                self.map_table[f+'_sum'].apply(lambda x: fr.Fraction(x, max_sum))
            self.map_table.astype({f+'_fr': fr.Fraction})

        # transform score on each field
        self.result_dict = dict()
        self.out_data = self.raw_data.copy(deep=True)
        for i, col in enumerate(self.cols):
            print('--- transform score field:[{}]'.format(col))

            self.raw_score_max = self.raw_data[col].max()
            self.raw_score_min = self.raw_data[col].min()

            # get formula and save
            if self.model_name in ['hainan', 'hainan2']:
                self.__get_formula_hainan(col)
            else:
                if not self.__get_formula(col):
                    print('getting formula fail !')
                    return

            # get field_plt in out_data
            print('   calculate: data[{0}] => {0}_plt'.format(col))
            self.out_data.loc[:, (col + '_plt')] = \
                self.out_data[col].apply(
                    lambda x: self.get_plt_score_from_formula_fraction(col, x))

            if self.out_decimal_digits == 0:
                self.out_data[col] = self.out_data[col].astype('int')
                self.out_data[col+'_plt'] = self.out_data[col+'_plt'].astype('int')

        # create col_plt in map_table
        df_map = self.map_table
        for col in self.cols:
            print('   calculate: map_table[{0}] => [{0}_plt]'.format(col))
            col_name = col + '_plt'
            df_map.loc[:, col_name] = df_map['seg'].apply(
                lambda x: self.get_plt_score_from_formula_fraction(col, x))
            if self.out_decimal_digits == 0:
                df_map[col_name] = df_map[col_name].astype('int')

        # make report doc
        self.make_report_doc()

        print('='*100)
        print('stm-run end, elapsed-time:', time.time() - stime)

    # run end

    # -----------------------------------------------------------------------------------
    # formula-1
    # y = a*x + b
    # a = (y2-y1)/(x2-x1)
    # b = -x1/(x2-x1) + y1
    def get_plt_score_from_formula_ax_b(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                return round45r(cf[0][0] * x + cf[0][1])
        return -1

    # -----------------------------------------------------------------------------------
    # formula-2
    # y = a*(x - b) + c
    # a = (y2-y1)/(x2-x1)
    # b = x1
    # c = y1
    def get_plt_score_from_formula_ax_b_c(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                v = (cf[1][1]-cf[1][0])
                if v == 0:
                    return round45r(cf[0][1])
                a = (cf[2][1]-cf[2][0])/v
                b = cf[1][0]
                c = cf[2][0]
                return round45r(a * (x - b) + c)
        return -1

    # -----------------------------------------------------------------------------------
    # formula-3 new, recommend to use,  int/int to float
    # original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
    # variant:  y = (a*x + b) / c
    #           a=(y2-y1)
    #           b=y1x2-y2x1
    #           c=(x2-x1)
    def get_plt_score_from_formula_fraction(self, field, x):
        if x > self.raw_score_max:
            return self.out_score_max
        if x < self.raw_score_min:
            return self.out_score_min
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                a = (cf[2][1]-cf[2][0])
                b = cf[2][0]*cf[1][1] - cf[2][1]*cf[1][0]
                c = (cf[1][1]-cf[1][0])
                if c == 0:  # x1 == x2: use mode_seg_degraded: max, min, mean(y1, y2)
                    if self.strategy_dict['mode_seg_degraded'] == 'max':
                        return max(cf[2])
                    elif self.strategy_dict['mode_seg_degraded'] == 'min':
                        return min(cf[2])
                    elif self.strategy_dict['mode_seg_degraded'] == 'mean':
                        return round45r(np.mean(cf[2]))
                    else:
                        return -1
                return round45r((a*x + b)/c)
        return -1

    # formula hainan
    # y = x for x in [x, x]
    # coeff: (a=0, b=x), (x, x), (y, y))
    # len(ratio_list) = len(map_table['seg'])
    def __get_formula_hainan(self, col):
        self.result_raw_endpoints = [x for x in self.map_table['seg']]
        self.map_table.loc[:, col+'_plt'] = -1
        coeff_dict = dict()
        result_ratio = []
        ep = 10**-8     # used to judge zero score(s==0) or equal to(s1==s2)
        _mode_order = self.strategy_dict['mode_score_order']
        _max = self.out_score_max if _mode_order in ['descending', 'd'] else self.out_score_min
        _step = -1 if _mode_order in ['descending', 'd'] else 1
        for ri, row in self.map_table.iterrows():
            x = row['seg']
            # processing-0: raw score == 0
            if abs(x) < ep:
                _mode_zero = self.strategy_dict['mode_score_zero']
                if 'ignore' in _mode_zero:
                    pass
                elif 'map_to_min' in _mode_zero:
                    if _mode_order in ['ascending', 'a']:
                        y = self.out_score_min
                        row[col+'_plt'] = y
                        coeff_dict.update({ri: [(0, y), (x, x), (y, y)]})
                        result_ratio.append(format(_p, '.6f'))
                    continue
                elif 'map_by_ratio' in _mode_zero:
                    pass
            for si, sr in enumerate(self.raw_score_ratio_cum):
                _p = row[col+'_percent']
                if (abs(_p - sr) < ep) or (_p < sr):
                    # strategies
                    _mode_zero = self.strategy_dict['mode_ratio_seek']
                    y = -1
                    if (abs(_p - sr) < ep) or (si == 0):
                        y = _max + si*_step
                    elif _mode_zero == 'upper_min':
                        y = _max + si*_step
                    elif _mode_zero == 'lower_max':
                        if si > 0:
                            y = _max + (si - 1)*_step
                        else:
                            y = _max + si*_step
                    elif 'near' in _mode_zero:
                        if abs(_p-sr) < abs(_p-self.raw_score_ratio_cum[si-1]):
                            y = _max - si
                        elif abs(_p-sr) > abs(_p-self.raw_score_ratio_cum[si-1]):
                            y = _max + (si - 1)*_step
                        else:
                            if 'near_max' in _mode_zero:
                                y = _max + si*_step
                            else:
                                y = _max + (si - 1)*_step
                    row[col+'_plt'] = y
                    coeff_dict.update({ri: [(0, y), (x, x), (y, y)]})
                    result_ratio.append(format(_p, '.6f'))
                    break
        self.result_formula_coeff = coeff_dict
        self.result_dict[col] = {'raw_score_points': self.result_raw_endpoints,
                                 'coeff': coeff_dict,
                                 'formula': ''}
        self.result_ratio_dict[col] = result_ratio

    def __get_formula(self, field):
        # --step 1
        # claculate raw_score_endpoints
        print('   get input score endpoints ...')
        points_list = self.__get_formula_raw_seg_list(field=field)
        self.result_raw_endpoints = points_list
        if len(points_list) == 0:
            return False
        # --step 2
        # calculate Coefficients
        self.__get_formula_coeff()
        self.result_dict[field] = {
            'raw_score_points': copy.deepcopy(self.result_raw_endpoints),
            'coeff': copy.deepcopy(self.result_formula_coeff),
            'formulas': copy.deepcopy(self.result_formula_text_list)
            }
        return True

    # -----------------------------------------------------------------------------------
    # formula-1: y = (y2-y1)/(x2 -x1)*(x - x1) + y1                   # a(x - b) + c
    #        -2:   = (y2-y1)/(x2 -x1)*x + (y1x2 - y2x1)/(x2 - x1)     # ax + b
    #        -3:   = [(y2-y1)*x + y1x2 - y2x1]/(x2 - x1)              # (ax + b) / c ; int / int
    def __get_formula_coeff(self):

        # create raw score segments list
        x_points = self.result_raw_endpoints
        step = 1 if self.strategy_dict['mode_score_order'] in ['ascending', 'a'] else -1
        x_list = [(int(p[0])+(step if i > 0 else 0), int(p[1]))
                  for i, p in enumerate(zip(x_points[:-1], x_points[1:]))]
        # 3-problems: minus score,
        #             less than min,
        #             no ratio interval(not found because of last too large ratio!)
        x_list = [(-1, -1)
                  if p[0] < 0 or min(p) < self.raw_score_min or (p[0]-p[1])*step > 0
                  else p
                  for p in x_list]

        # calculate coefficient
        y_list = self.out_score_points
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            v = x[1] - x[0]
            if v == 0:
                a = 0
                # mode_seg_degraded
                _mode_seg_degraded = self.strategy_dict['mode_seg_degraded']
                if _mode_seg_degraded == 'max':     # x1 == x2 : y = max(y1, y2)
                    b = max(y)
                elif _mode_seg_degraded == 'min':   # x1 == x2 : y = min(y1, y2)
                    b = min(y)
                else:                                   # x1 == x2 : y = mean(y1, y2)
                    b = np.mean(y)
            else:
                a = (y[1]-y[0])/v                   # (y2 - y1) / (x2 - x1)
                b = (y[0]*x[1]-y[1]*x[0])/v         # (y1x2 - y2x1) / (x2 - x1)
            self.result_formula_coeff.update({i: [(a, b), x, y]})
        return True

    # new at 2019-09-09
    def __get_formula_raw_seg_list(self, field):
        result_ratio = []
        if self.strategy_dict['mode_score_min'] == 'map_by_ratio':
            _score_min = self.raw_data[field].min()
            _score_max = self.raw_data[field].max()
        else:
            _score_min = self.raw_score_min
            _score_max = self.raw_score_max
        _mode_cumu = self.strategy_dict['mode_ratio_cumu']
        _mode_order = self.strategy_dict['mode_score_order']
        _ratio_cum_list = self.raw_score_ratio_cum

        # start points for raw score segments
        raw_score_start = _score_min if _mode_order in ['a', 'ascending'] else _score_max
        result_raw_seg_list = [raw_score_start]

        # ratio: predefined,  percent: computed from data
        last_ratio = 0
        last_percent = 0
        for i, cumu_ratio in enumerate(_ratio_cum_list):
            this_seg_ratio = cumu_ratio-last_ratio
            dest_percent = cumu_ratio if _mode_cumu == 'no' else this_seg_ratio+last_percent

            # It is defined in model: cumu_ratio[-1]==1
            # if i == len(_ratio_cum_list)-1:
            #     dest_percent = 1.0

            # seek first endpoint and real cumulative percent of each segment from map_table
            this_seg_endpoint, this_seg_percent = self.get_seg_from_map_table(field, dest_percent)

            # save last segment endpoint and percent
            last_ratio = cumu_ratio
            last_percent = this_seg_percent

            # save to result ratio
            result_ratio.append('{:.6f}'.format(this_seg_percent))

            # set result endpoints (linked, share)
            if cumu_ratio == _ratio_cum_list[-1]:       # last ratio segment
                # if last endpoit is at bottom, this is set to -1
                if result_raw_seg_list[-1] in [self.raw_score_min, self.raw_score_max]:
                    this_seg_endpoint = -1
            result_raw_seg_list.append(this_seg_endpoint)

            # print(this_seg_endpoint)
            print('   <{}> ratio: [spec:{:.4f}  locate:{:.4f}  result:{:.4f}] => '
                  'intervals:(raw:[{:3.0f}, {:3.0f}]  out:[{:3.0f}, {:3.0f}])'.
                  format(i+1,
                         cumu_ratio,
                         dest_percent,
                         this_seg_percent,
                         result_raw_seg_list[-2] if i == 0 else
                            (result_raw_seg_list[-2]-1 if this_seg_endpoint >= self.raw_score_min else -1),
                         this_seg_endpoint,
                         self.out_score_points[i][0],
                         self.out_score_points[i][1]
                         )
                  )

        self.result_ratio_dict[field] = result_ratio
        return result_raw_seg_list

    # new at 2019-09-09
    def get_seg_from_map_table(self, field, dest_ratio):

        _mode = self.strategy_dict['mode_ratio_seek']
        map_table = self.map_table
        _tiny = 10**-8
        _seg = -1
        _percent = -1
        last_percent = -1
        last_seg = None
        last_diff = 1000
        _use_last = False
        for index, row in map_table.iterrows():
            _percent = row[field+'_percent']
            _seg = row['seg']
            _diff = abs(_percent - dest_ratio)

            # at table bottom or lowest score, use_current
            # may process strategy later in seg_list for score_min = 'real/zero'
            if (index == map_table.index.max()) or (_percent >= 1):
                break

            # reach bigger than or equal to ratio
            # no effect on stratedy: mode_score_empty
            if _percent >= dest_ratio:
                # at top row
                if last_seg is None:
                    break
                # dealing with strategies
                if 'near' in _mode:
                    # (distances are same, and _mode is near_min) or (last is near)
                    if ((abs(_diff-last_diff) < _tiny) and ('near_min' in _mode)) or \
                       (_diff > last_diff):
                        _use_last = True
                elif _mode == 'lower_max':
                    if abs(_percent-dest_ratio) > _tiny:
                        _use_last = True
                elif _mode == 'upper_min':
                    pass
                else:
                    raise ValueError
                break
            last_seg = _seg
            last_diff = _diff
            last_percent = _percent
        if _use_last:
            return last_seg, last_percent
        return _seg, _percent

    @classmethod
    def get_seg_from_fr(cls, 
                        mapdf: pd.DataFrame,
                        field,
                        ratio):
        # comments:
        #   use limit_denominator in Fraction
        #   because of the error in pandas.field(Fraction) is not valid
        _r = fr.Fraction(ratio).limit_denominator(1000000)
        last_fr = -1
        last_seg = -1
        start = 0
        for row_id, row in mapdf.iterrows():
            this_fr = row[field+'_fr']
            this_seg = row['seg']
            if (_r <= this_fr) or (start == len(mapdf)):
                if (start == 0) or (_r == this_fr):
                    return (this_seg, this_fr, _r-last_fr), (this_seg, this_fr, this_fr-_r)
                return (last_seg, last_fr, _r-last_fr), (this_seg, this_fr, this_fr-_r)
            last_fr = this_fr
            last_seg = this_seg
            start += 1

    # create report and col_plt in map_table
    def make_report_doc(self):
        self.out_report_doc = 'Transform Model: [{}]   {}\n'.\
            format(self.model_name, time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime()))
        self.out_report_doc += '---'*40 + '\n'
        self.out_report_doc += format('strategies: ', '>23') + '\n'

        for k in stm_strategies_dict:
            self.out_report_doc += ' ' * 23 + '{:<32s} {}'. \
                format(k + ' = ' + self.strategy_dict[k],
                       stm_strategies_dict[k]) + '\n'
        self.out_report_doc += '---'*40 + '\n'
        for col in self.cols:
            print('   create report ...')
            self.out_report_doc += self.__get_report_doc(col)

    def __get_report_doc(self, field=''):
        score_dict = {x: y for x, y in zip(self.map_table['seg'], self.map_table[field+'_count'])}
        p = 0 if self.strategy_dict['mode_score_order'] in ['ascending', 'a'] else 1
        self.result_formula_text_list = []
        _fi = 1
        for k in self.result_dict[field]['coeff']:
            formula = self.result_dict[field]['coeff'][k]
            _break = True
            _step = 1 if formula[1][0] < formula[1][1] else -1
            for _score in range(formula[1][0], formula[1][1]+_step, _step):
                if score_dict.get(_score, -1) > 0:
                    _break = False
                    break
            if _break:
                continue
            if formula[1][0] < 0 or formula[1][0] < formula[1][1]:
                self.result_formula_text_list += ['(seg-{:3d}) ******'.format(_fi)]
                continue
            if formula[0][0] > 0:
                self.result_formula_text_list += \
                    ['(seg-{0:3d}) y = {1:0.8f}*(x-{2:2d}) + {3:2d}'.
                     format(_fi, formula[0][0], formula[1][p], formula[2][p])]
            elif formula[0][0] == 0:
                if formula[2][0] != formula[2][1]:
                    self.result_formula_text_list += \
                        ['(seg-{0:3d}) y = {1:0.8f}*(x-{2:3d}) + {3}({4:3d}, {5:3d})'.
                         format(_fi,
                                formula[0][0], formula[1][p],
                                self.strategy_dict['mode_seg_degraded'],
                                formula[2][0], formula[2][1])
                         ]
                else:
                    self.result_formula_text_list += \
                        ['(seg-{0:3d}) y = 1.0*(x-{2:3d}) + {3:3d}'.
                         format(_fi,
                                formula[0][0],
                                formula[1][p],
                                formula[2][0])
                         ]
            _fi += 1

        # report start
        # tiltle
        field_title = '<< score field: [{}] >>\n' + '- -'*40 + '\n'
        _out_report_doc = field_title.format(field)

        # calculating for ratio and segment
        plist = self.raw_score_ratio_cum
        _out_report_doc += '  raw score seg ratio: {}\n'.\
            format([format(plist[j]-plist[j-1] if j > 0 else plist[0], '0.6f')
                    for j in range(len(plist))])
        _out_report_doc += '  raw score cum ratio: {}\n'.\
            format([format(x, '0.6f') for x in self.raw_score_ratio_cum])
        _out_report_doc += '  raw score set ratio: {}\n'.\
            format(self.result_ratio_dict[field])
        _raw_seg_list = [x[1] for x in self.result_dict[field]['coeff'].values()]
        if len(_raw_seg_list) > 20:     # for hainan too many segs(801) and single point seg
            _raw_seg_list = [x[0] if x[0] == x[1] else x for x in _raw_seg_list]
        _out_report_doc += '  raw score endpoints: {}\n'.\
            format(_raw_seg_list)
        _out_seg_list = [x[2] for x in self.result_dict[field]['coeff'].values()]
        if len(_raw_seg_list) > 20:     # for hainan too many segs(801) and single point seg
            _out_seg_list = [x[0] if x[0] == x[1] else x for x in _out_seg_list]
        _out_report_doc += '  out score endpoints: {}\n'.\
            format(_out_seg_list)

        # transforming formulas
        _out_report_doc += '- -'*40 + '\n'
        for i, col in enumerate(self.result_formula_text_list):
            if i == 0:
                _out_report_doc += 'transforming formulas:\n'
            _out_report_doc += '                       {}\n'.format(col)

        # statistics for raw and out score
        _out_report_doc += '- -'*40 + '\n'
        _out_report_doc += format('statistics:', '>22s')

        # raw score data describing
        _max, _min, _mean, _median, _mode, _std, _skew, _kurt = \
            self.raw_data[field].max(),\
            self.raw_data[field].min(),\
            self.raw_data[field].mean(),\
            self.raw_data[field].median(), \
            self.raw_data[field].mode()[0], \
            self.raw_data[field].std(),\
            self.raw_data[field].skew(),\
            sts.kurtosis(self.raw_data[field], fisher=False)
        _out_report_doc += ' raw: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                              format(_max, _min, _mean, _median, _mode)
        _out_report_doc += ' '*28 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                              format(_std, _std/_mean, _max-_min, _skew, _kurt)
        # _count_zero = self.map_table.query(field+'_count==0')['seg'].values
        _count_non_zero = self.map_table.groupby('seg')[[field+'_count']].sum().query(field+'_count>0').index
        _count_zero = [x for x in range(self.raw_score_range[0], self.raw_score_range[1]+1)
                       if x not in _count_non_zero]
        _out_report_doc += ' '*28 + 'empty_value={}\n' .\
                              format(_count_zero)

        # out score data describing
        _max, _min, _mean, _median, _mode, _std, _skew, _kurt = \
            self.out_data[field+'_plt'].max(),\
            self.out_data[field+'_plt'].min(),\
            self.out_data[field+'_plt'].mean(),\
            self.out_data[field+'_plt'].median(), \
            self.out_data[field+'_plt'].mode()[0],\
            self.out_data[field+'_plt'].std(),\
            self.out_data[field+'_plt'].skew(), \
            sts.kurtosis(self.out_data[field+'_plt'], fisher=False)
        _out_report_doc += ' '*23 + 'out: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                              format(_max, _min, _mean, _median, _mode)
        _out_report_doc += ' '*28 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                              format(_std, _std/_mean, _max-_min, _skew, _kurt)
        # _count_zero = self.map_table.query(field+'_count==0')[field+'_plt'].values
        _count_non_zero = self.map_table.groupby(field+'_plt')[[field+'_count']].sum().query(field+'_count>0').index
        _count_zero = [x for x in range(self.out_score_min, self.out_score_max+1) 
                       if x not in _count_non_zero]
        _out_report_doc += ' '*28 + 'empty_value={}\n' .\
                              format(_count_zero)

        # differece between raw and out score
        _out_report_doc += '- -'*40 + '\n'
        _diff_raw_out = self.out_data[field+'_plt']-self.out_data[field]
        _out_report_doc += ' score shift(out-raw):' \
                              ' shift_max={:3.1f}' \
                              '  shift_min={:3.1f}' \
                              '  shift_down_percent={:.2f}%\n'.\
                              format(
                                    max(_diff_raw_out),
                                    min(_diff_raw_out),
                                    _diff_raw_out[_diff_raw_out < 0].count()/_diff_raw_out.count()*100
                              )
        _diff_list = []
        for coeff in self.result_formula_coeff.values():
            rseg = coeff[1]
            oseg = coeff[2]
            a = coeff[0][0]
            b = coeff[0][1]
            # print(rseg, oseg)
            if rseg[0] >= oseg[0]:
                if rseg[1] > oseg[1]:
                    _diff_list.append(rseg)
            if (rseg[0] > oseg[0]) and (rseg[1] <= oseg[1]):
                _diff_list.append((int(rseg[0]), int(round45r(b/(1-a)))))
            if (rseg[0] < oseg[0]) and (rseg[1] >= oseg[1]):
                _diff_list.append((int(round45r(b/(1-a), 0)), int(rseg[1])))
        _out_report_doc += '   shift down segment: ' + str(_diff_list) + ' => '
        while True:
            _diff_loop = False
            for i in range(len(_diff_list)-1):
                if abs(_diff_list[i][1]-_diff_list[i+1][0]) == 1:
                    _diff_list[i] = (_diff_list[i][0], _diff_list[i+1][1])
                    _diff_list.pop(i+1)
                    _diff_loop = True
                    break
            if not _diff_loop:
                break
        _diff_list = [x for x in _diff_list if x[0] != x[1]]
        _out_report_doc += str(_diff_list) + '\n'
        _out_report_doc += '---'*40 + '\n'

        return _out_report_doc

    def report(self):
        print(self.out_report_doc)

    def plot(self, mode='model'):
        if mode not in ['raw', 'out', 'model', 'shift', 'dist', 'bar', 'diff', 'normtest']:
            print('valid mode is: raw, out, model,shift, dist, bar, diff, normtest')
            return
        if mode in 'shift, model':
            # mode: model describe the differrence of input and output score.
            self.__plot_model()
        elif mode in 'dist':
            self.__plot_dist_seaborn()
        elif mode in 'bar':
            self.__plot_bar()
        elif mode in 'diff':
            self.__plot_diff()
        elif mode in 'normtest':
            self.__plot_norm_test()
        elif not super(PltScore, self).plot(mode):
            print('\"{}\" is invalid'.format(mode))

    def __plot_norm_test(self):
        self.norm_test = dict()
        for col in self.cols:
            _len = self.map_table[col+'_count'].sum()
            x1 = sorted(self.out_data[col])
            x2 = sorted(self.out_data[col+'_plt'])
            y = [(_i-0.375)/(_len+0.25) for _i in range(1, _len+1)]
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+': norm test')
            ax.plot(x1, y, 'o-', label='score:' + col)
            ax.plot(x2, y, 'o-', label='score:' + col)

    def __plot_diff(self):
        x = [int(x) for x in self.map_table['seg']][::-1]   # np.arange(self.raw_score_max+1)
        raw_label = [str(x) for x in self.map_table['seg']][::-1]
        for f in self.cols:
            raw_data = [v if self.map_table.query('seg=='+str(v))[f+'_count'].values[0] > 0 else 0 for v in x]
            out_data = list(self.map_table[f + '_plt'])[::-1]
            out_data = [out if raw > 0 else 0 for raw, out in zip(raw_data, out_data)]

            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: diffrence between raw and out')
            ax.set_xticks(x)
            ax.set_xticklabels(raw_label)
            width = 0.4
            bar_wid = [p - width/2 for p in x]
            rects1 = ax.bar(bar_wid, raw_data, width, label=f)
            bar_wid = [p + width/2 for p in x]
            rects2 = ax.bar(bar_wid, out_data, width, label=f+'_plt')

            """Attach a text label above each bar in *rects*, displaying its height."""
            for i, rects in enumerate([rects1, rects2]):
                for rect in rects:
                    if i == 0:
                        notes = rect.get_height()
                    else:
                        if rect.get_height() > 0:
                            notes = rect.get_height() - rect.get_x()
                        else:
                            notes = 0
                    height = rect.get_height()
                    ax.annotate('{}'.format(int(notes)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            ax.legend(loc='upper left', shadow=True, fontsize='x-large')
            fig.tight_layout()
            plot.show()

    def __plot_bar(self):
        raw_label = [str(x) for x in range(self.out_score_max+1)]
        x_data = list(range(self.out_score_max+1))
        seg_list = list(self.map_table.seg)
        for f in self.cols:
            raw_data = [self.map_table.query('seg=='+str(xv))[f+'_count'].values[0]
                        if xv in seg_list else 0
                        for xv in x_data]
            out_ = self.out_data.groupby(f+'_plt').count()[f]    # .sort_index(ascending=False)
            out_data = [out_[int(v)] if int(v) in out_.index else 0 for v in raw_label]
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: bar graph')
            ax.set_xticks(x_data)
            ax.set_xticklabels(raw_label)
            width = 0.4
            bar_wid = [p - width/2 for p in x_data]
            raw_bar = ax.bar(bar_wid, raw_data, width, label=f)
            bar_wid = [p + width/2 for p in x_data]
            out_bar = ax.bar(bar_wid, out_data, width, label=f+'_plt')
            for bars in [raw_bar, out_bar]:
                for _bar in bars:
                    height = _bar.get_height()
                    ax.annotate('{}'.format(int(height)),
                                xy=(_bar.get_x() + _bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            fig.tight_layout()
            plot.show()

    def __plot_dist(self):
        def plot_hist_fit(field, _label):
            x_data = self.out_data[field]
            _mu = np.mean(x_data)
            _std = np.std(x_data)
            count, bins, patches = ax.hist(x_data, 35)
            x_fit = ((1 / (np.sqrt(2 * np.pi) * _std)) * np.exp(-0.5 * (1 / _std * (bins - _mu))**2))
            x_fit = x_fit * max(count)/max(x_fit)
            _color = 'y--' if '_plt' in field else 'g--'
            ax.plot(bins, x_fit, _color, label=_label)
            ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            # print(field, len(count), sum(count), count)
        for f in self.cols:
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: distribution garph')
            # fit raw score distribution
            plot_hist_fit(f, 'raw score')
            # fit out score distribution
            plot_hist_fit(f+'_plt', 'out score')
        plot.show()

    def __plot_dist_seaborn(self):
        for f in self.cols:
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: distribution garph')
            sbn.kdeplot(self.out_data[f], shade=True)
            sbn.kdeplot(self.out_data[f+'_plt'], shade=True)

    def __plot_model(self):
        # 分段线性转换模型
        plot.rcParams['font.sans-serif'] = ['SimHei']
        plot.rcParams.update({'font.size': 8})
        for i, col in enumerate(self.cols):
            result = self.result_dict[col]
            raw_points = result['raw_score_points']
            in_max = max(raw_points)
            ou_min = min([min(p) for p in self.out_score_points])
            ou_max = max([max(p) for p in self.out_score_points])

            plot.figure(col+'_plt')
            plot.rcParams.update({'font.size': 10})
            plot.title(u'转换模型({})'.format(col))
            plot.xlim(min(raw_points), max(raw_points))
            plot.ylim(ou_min, ou_max)
            plot.xlabel(u'\n原始分数')
            plot.ylabel(u'转换分数')
            plot.xticks([])
            plot.yticks([])

            formula = self.result_dict[col]['coeff']
            for cfi, cf in enumerate(formula.values()):
                _score_order = self.strategy_dict['mode_score_order']
                x = cf[1] if _score_order in ['ascending', 'a'] else cf[1][::-1]
                y = cf[2] if _score_order in ['ascending', 'a'] else cf[2][::-1]
                plot.plot(x, y, linewidth=3)
                for j in [0, 1]:
                    plot.plot([x[j], x[j]], [0, y[j]], '--', linewidth=2)
                    plot.plot([0, x[j]], [y[j], y[j]], '--', linewidth=2)
                for j, xx in enumerate(x):
                    plot.text(xx-1 if j == 1 else xx, ou_min-2, '{}'.format(int(xx)))
                for j, yy in enumerate(y):
                    plot.text(1, yy-2 if j == 1 else yy+1, '{}'.format(int(yy)))

            # darw y = x for showing score shift
            plot.plot((0, in_max), (0, in_max), 'm--', linewidth=3, markersize=3)

        plot.show()
        return

    def report_map_table(self):
        fs_list = ['seg']
        for ffs in self.cols:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_plt']
        print(self.map_table[fs_list])


class Zscore(ScoreTransformModel):
    """
    transform raw score to Z-score according to percent position on normal cdf
    input data: raw_data = raw score dataframe
    set parameters: stdNum = standard error numbers
    output data: out_data = result score with raw score field name + '_z'
    """
    # HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self):
        super(Zscore, self).__init__('zscore')
        self.model_name = 'zscore'

        # input data
        self.raw_data = None
        self.cols = None

        # model parameters
        self.std_num = 4
        self.norm_table_len = 10000
        self.raw_score_max = 150
        self.raw_score_min = 0
        self.out_data_decimal = 0
        self.out_score_number = 100
        self.mode_ratio_seek = 'near'
        self.norm_table = array.array('d', [sts.norm.cdf(-self.std_num * (1 - 2 * x / (self.norm_table_len - 1)))
                                            for x in range(self.norm_table_len)]
                                      )
        # result data
        self.map_table = None
        self.out_data = None

    def set_data(self, raw_data=None, cols=None):
        self.raw_data = raw_data
        self.cols = cols

    def set_para(self,
                 std_num=4,
                 raw_score_range=(0, 100),
                 mode_ratio_seek='near',
                 out_decimal=8
                 ):
        self.std_num = std_num
        self.raw_score_max = raw_score_range[1]
        self.raw_score_min = raw_score_range[0]
        self.mode_ratio_seek = mode_ratio_seek
        self.out_data_decimal = out_decimal

    def check_parameter(self):
        if self.raw_score_max <= self.raw_score_min:
            print('error: max raw score is less than min raw score!')
            return False
        if self.std_num <= 0:
            print('error: std number {} is error!'.format(self.std_num))
            return False
        return True

    # Zscore run
    def run(self):
        # check data and parameter in super
        if not super(Zscore, self).run():
            return
        print('start run...')
        st = time.clock()
        self.out_data = self.raw_data.copy()
        self.map_table = self.get_map_table(
            self.out_data,
            self.raw_score_max,
            self.raw_score_min,
            self.cols,
            seg_order='a')
        for col in self.cols:
            print('calc zscore on field: {}...'.format(col))
            self.map_table[col+'_zscore'] = self.get_zscore(self.map_table[col+'_percent'])
            map_dict = {rscore: zscore for rscore, zscore in
                        zip(self.map_table['seg'], self.map_table[col + '_zscore'])}
            self.out_data.loc[:, col + '_zscore'] = \
                self.out_data[col].apply(lambda x: map_dict.get(x, -999))
        print('zscore finished with {} consumed'.format(round(time.clock()-st, 2)))

    # new method for uniform algorithm with strategies
    def get_zscore(self, percent_list):
        # z_list = [None for _ in percent_list]
        z_array = array.array('d', range(len(percent_list)))
        _len = self.norm_table_len
        for i, _p in enumerate(percent_list):
            pos = bst.bisect(self.norm_table, _p)
            z_array[i] = 2*(pos - _len/2) / _len * self.std_num
        return z_array

    @staticmethod
    def get_map_table(df, maxscore, minscore, cols, seg_order='a'):
        seg = SegTable()
        seg.set_data(df, cols)
        seg.set_para(segmax=maxscore, segmin=minscore, segsort=seg_order)
        seg.run()
        return seg.out_data

    def report(self):
        if type(self.out_data) == pd.DataFrame:
            print('output score desc:\n', self.out_data.describe())
        else:
            print('output score data is not ready!')
        print('data fields in raw_score:{}'.format(self.cols))
        print('para:')
        print('\tzscore stadard diff numbers:{}'.format(self.std_num))
        print('\tmax score in raw score:{}'.format(self.raw_score_max))
        print('\tmin score in raw score:{}'.format(self.raw_score_min))

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super(Zscore, self).plot(mode)
        else:
            print('not support this mode!')

# === Zscore model end ===


class Tscore(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模,平均数为50,标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出,是为了纪念推孟和桑代克
    对智力测验,尤其是提出智商这一概念所作出的巨大贡献。
    通过调整t_score_mean, t_score_std, 也可以进行其它标准分数转换，
    如100-900分的标准分数转换。
    本模型使用百分位-累计分布校准的方式计算转换分数。
    '''

    def __init__(self):
        super(Tscore, self).__init__('t')
        # self.model_name = 't'

        self.raw_score_max = 100
        self.raw_score_min = 0
        self.t_score_std = 10
        self.t_score_mean = 50
        self.t_score_stdnum = 4

        self.out_data_decimal = 0
        self.zscore_decimal = 8

        self.map_table = None

    def set_data(self, raw_data=None, cols=None):
        self.raw_data = raw_data
        self.cols = cols

    def set_para(self, 
                 raw_score_range=(0, 100),
                 t_score_mean=50,
                 t_score_std=10,
                 t_score_stdnum=4,
                 out_decimal=0):
        self.raw_score_max = raw_score_range[1]
        self.raw_score_min = raw_score_range[0]
        self.t_score_mean = t_score_mean
        self.t_score_std = t_score_std
        self.t_score_stdnum = t_score_stdnum
        self.out_data_decimal = out_decimal

    def run(self):
        zm = Zscore()
        zm.set_data(self.raw_data, self.cols)
        zm.set_para(std_num=self.t_score_stdnum,
                    raw_score_range=(self.raw_score_min, self.raw_score_max),
                    out_decimal=self.zscore_decimal
                    )
        zm.run()
        self.out_data = zm.out_data
        namelist = self.out_data.columns

        def formula(x):
            return round45r(x * self.t_score_std + self.t_score_mean, self.out_data_decimal)

        for sf in namelist:
            if '_zscore' in sf:
                new_sf = sf.replace('_zscore', '_tscore')
                self.out_data.loc[:, new_sf] = self.out_data[sf].apply(formula)
        self.map_table = zm.map_table

    def report(self):
        print('T-score by normal table transform report')
        print('-' * 50)
        if type(self.raw_data) == pd.DataFrame:
            print('raw score desc:')
            print('    fields:', self.cols)
            print(self.raw_data[self.cols].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.out_data) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.cols]
            print('T-score desc:')
            print('    fields:', out_fields)
            print(self.out_data[out_fields].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in raw_score:{}'.format(self.cols))
        print('-' * 50)
        print('para:')
        print('\tzscore stadard deviation numbers:{}'.format(self.t_score_std))
        print('\tmax score in raw score:{}'.format(self.raw_score_max))
        print('\tmin score in raw score:{}'.format(self.raw_score_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super(Tscore, self).plot(mode)


class TscoreLinear(ScoreTransformModel):
    """Get Zscore by linear formula: (x-mean)/std"""
    def __init__(self):
        super(TscoreLinear, self).__init__('tzl')

        self.model_name = 'tzl'
        self.raw_score_max = 100
        self.raw_score_min = 0
        self.t_score_mean = 50
        self.t_score_std = 10
        self.t_score_stdnum = 4

    def set_data(self, raw_data=None, cols=None):
        self.raw_data = raw_data
        self.cols = cols

    def set_para(self,
                 raw_score_max=100,
                 raw_score_min=0,
                 t_score_std=10,
                 t_score_mean=50,
                 t_score_stdnum=4):
        self.raw_score_max = raw_score_max
        self.raw_score_min = raw_score_min
        self.t_score_mean = t_score_mean
        self.t_score_std = t_score_std
        self.t_score_stdnum = t_score_stdnum

    def check_data(self):
        super(TscoreLinear, self).check_data()
        return True

    def check_parameter(self):
        if self.raw_score_max <= self.raw_score_min:
            print('raw score max and min error!')
            return False
        if self.t_score_std <= 0 | self.t_score_stdnum <= 0:
            print('t_score std number error:std={}, stdnum={}'.format(self.t_score_std, self.t_score_stdnum))
            return False
        return True

    def run(self):
        super(TscoreLinear, self).run()
        self.out_data = self.raw_data
        for sf in self.cols:
            rmean, rstd = self.out_data[[sf]].describe().loc[['mean', 'std']].values[:, 0]
            self.out_data[sf + '_zscore'] = \
                self.out_data[sf].apply(
                    lambda x: min(max((x - rmean) / rstd, -self.t_score_stdnum), self.t_score_stdnum))
            self.out_data.loc[:, sf + '_tscore'] = \
                self.out_data[sf + '_zscore'].\
                apply(lambda x: x * self.t_score_std + self.t_score_mean)

    def report(self):
        print('TZ-score by linear transform report')
        print('-' * 50)
        if type(self.raw_data) == pd.DataFrame:
            print('raw score description:')
            print(self.raw_data[[f for f in self.cols]].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.out_data) == pd.DataFrame:
            print('raw,T,Z score description:')
            print(self.out_data[[f+'_tscore' for f in self.cols]].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in raw_score:{}'.format(self.cols))
        print('-' * 50)
        print('para:')
        print('\tzscore stadard deviation numbers:{}'.format(self.t_score_std))
        print('\tmax score in raw score:{}'.format(self.raw_score_max))
        print('\tmin score in raw score:{}'.format(self.raw_score_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super(TscoreLinear, self).plot(mode)


class GradeScoreTai(ScoreTransformModel):
    """
    Grade Score Model used by Taiwan College Admission Test Center
    top_group = df.sort_values(field,ascending=False).head(int(df.count(0)[field]*0.01))[[field]]
    high_grade_score = round(top_group[field].mean(), 4)
    intervals = [minscore, grade_level/grade_level_total_number], ..., [,high_grade]
    以原始分值切分，形成的分值相当于等距合并，粒度直接增加
    实质上失去了等级分数的意义
    本模型仍然存在高分区过度合并问题
    """

    def __init__(self):
        super(GradeScoreTai, self).__init__('grade')
        self.model_name = 'Taiwan'

        self.grade_num = 15
        self.raw_score_max = 100
        self.raw_score_min = 0
        self.max_ratio = 0.01   # 1%
        self.raw_data = pd.DataFrame()

        self.grade_no = [x for x in range(self.grade_num+1)]
        self.map_table = None
        self.grade_dist_dict = {}  # col: grade_list, from max to min
        self.out_data = pd.DataFrame()

    def set_data(self, raw_data=pd.DataFrame(), cols=None):
        if len(raw_data) > 0:
            self.raw_data = raw_data
        if isinstance(cols, list) or isinstance(cols, tuple):
            self.cols = cols

    def set_para(self,
                 raw_score_max=None,
                 raw_score_min=None,
                 grade_num=None,
                 ):
        if isinstance(raw_score_max, int):
            if len(self.cols) > 0:
                if raw_score_max >= max([max(self.raw_data[f]) for f in self.cols]):
                    self.raw_score_max = raw_score_max
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set col first!')
        if isinstance(raw_score_min, int):
            self.raw_score_min = raw_score_min
        if isinstance(grade_num, int):
            self.grade_num = grade_num
        self.grade_no = [x for x in range(self.grade_num+1)]

    def run(self):
        self.run_create_grade_dist_list()
        self.run_create_out_data()

    def run_create_grade_dist_list(self):
        # mode_ratio_seek = 'near'
        seg = SegTable()
        seg.set_para(segmax=self.raw_score_max,
                     segmin=self.raw_score_min,
                     segsort='d')
        seg.set_data(self.raw_data,
                     self.cols)
        seg.run()
        self.map_table = seg.out_data
        for fs in self.cols:
            lastpercent = 0
            lastseg = self.raw_score_max
            for ind, row in self.map_table.iterrows():
                curpercent = row[fs + '_percent']
                curseg = row['seg']
                if row[fs+'_percent'] > self.max_ratio:
                    if curpercent - self.max_ratio > self.max_ratio - lastpercent:
                        max_score = lastseg
                    else:
                        max_score = curseg
                    max_point = self.raw_data[self.raw_data[fs] >= max_score][fs].mean()
                    # print(fs, max_score, curseg, lastseg)
                    self.grade_dist_dict.update({fs: round45r(max_point/self.grade_num, 8)})
                    break
                lastpercent = curpercent
                lastseg = curseg

    def run_create_out_data(self):
        dt = copy.deepcopy(self.raw_data[self.cols])
        for fs in self.cols:
            dt.loc[:, fs+'_grade'] = dt[fs].apply(lambda x: self.run__get_grade_score(fs, x))
            dt2 = self.map_table
            dt2.loc[:, fs+'_grade'] = dt2['seg'].apply(lambda x: self.run__get_grade_score(fs, x))
            self.out_data = dt

    def run__get_grade_score(self, fs, x):
        if x == 0:
            return x
        grade_dist = self.grade_dist_dict[fs]
        for i in range(self.grade_num):
            minx = i * grade_dist
            maxx = (i+1) * grade_dist if i < self.grade_num-1 else self.raw_score_max
            if minx < x <= maxx:
                return i+1
        return -1

    def plot(self, mode='raw'):
        pass

    def report(self):
        print(self.out_data[[f+'_grade' for f in self.cols]].describe())

    def print_map_table(self):
        print(self.map_table)


# call SegTable.run() return instance of SegTable
def run_seg(
            raw_data: pd.DataFrame,
            cols: list,
            segmax=100,
            segmin=0,
            segsort='d',
            segstep=1,
            display=False,
            usealldata=False
            ):
    seg = SegTable()
    seg.set_data(
        raw_data=raw_data,
        cols=cols
    )
    seg.set_para(
        segmax=segmax,
        segmin=segmin,
        segstep=segstep,
        segsort=segsort,
        display=display,
        useseglist=usealldata
    )
    seg.run()
    return seg


# version 1.0.1 2018-09-24
class SegTable(object):
    """
    * 计算pandas.DataFrame中分数字段的分段人数表
    * segment table for score dataframe
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    * from 09-17-2017

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(raw_data:DataFrame, fs:list)
        raw_data: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        fs: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离，分段开始值，分数顺序，指定分段值列表， 使用指定分段列表，使用所有数据， 关闭计算过程显示信息
    set_para（segmax, segmin, segstep, segstart, segsort, seglist, useseglist, usealldata, display）
        segmax: int, maxvalue for segment, default=150
                输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。
                输出分段表中分数段的最小值
        segstep: int, grades for segment value, default=1
                分段间隔，用于生成n-分段表（五分一段的分段表）
        segstart:int, start seg score to count
                进行分段计算的起始值
        segsort: str, 'a' for ascending order or 'd' for descending order, default='d' (seg order on descending)
                输出结果中分段值得排序方式，d: 从大到小， a：从小到大
                排序模式的设置影响累计数和百分比的意义。
        seglist: list, used to create set value
                 使用给定的列表产生分段表，列表中为分段点值
        useseglist: bool, use or not use seglist to create seg value
                 是否使用给定列表产生分段值
        usealldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        display: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息
    out_data: 输出分段数据
            seg: seg value
        [field]: field name in fs
        [field]_count: number at the seg
        [field]_sum: cumsum number at the seg
        [field]_percent: percentage at the seg
        [field]_count[step]: count field for step != 1
        [field]_list: count field for assigned seglist when use seglist
    运行，产生输出数据, calculate and create output data
    run()

    应用举例
    example:
        import pyex_seg as sg
        seg = SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_para(segmax=100, segmin=1, segstep=1, segsort='d', usealldata=True, display=True)
        seg.run()
        seg.plot()
        print(seg.out_data.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          fs type is digit, for example: int or float

        3)可以单独设置数据(raw_data),字段列表（fs),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.col = ['score_1', 'score_2'];
              seg.segmax = 120
          重新设置后需要运行才能更新输出数据ouput_data, 即调用run()
          便于在计算期间调整模型。
          by usting property mode, rawdata, scorefields, para can be setted individually
        4) 当设置大于1分的分段分值X时， 会在结果DataFrame中生成一个字段[segfiled]_countX，改字段中不需要计算的分段
          值设为-1。
          when segstep > 1, will create field [segfield]_countX, X=str(segstep), no used value set to -1 in this field
    """

    def __init__(self):
        # raw data
        self.__raw_dataframe = None
        self.__cols = []

        # parameter for model
        self.__segList = []
        self.__useseglist = False
        self.__segStart = 100
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'd'
        self.__usealldata = True
        self.__display = True
        self.__percent_decimal = 10

        # result data
        self.__out_dataframe = None

        # run status
        self.__run_completed = False

    @property
    def out_data(self):
        return self.__out_dataframe

    @property
    def raw_data(self):
        return self.__raw_dataframe

    @raw_data.setter
    def raw_data(self, df):
        self.__raw_dataframe = df

    @property
    def cols(self):
        return self.__cols

    @cols.setter
    def cols(self, cols):
        self.__cols = cols

    @property
    def seglist(self):
        return self.__segList

    @seglist.setter
    def seglist(self, seglist):
        self.__segList = seglist

    @property
    def useseglist(self):
        return self.__useseglist

    @useseglist.setter
    def useseglist(self, useseglist):
        self.__useseglist = useseglist

    @property
    def segstart(self):
        return self.__segStart

    @segstart.setter
    def segstart(self, segstart):
        self.__segStart = segstart

    @property
    def segmax(self):
        return self.__segMax

    @segmax.setter
    def segmax(self, segvalue):
        self.__segMax = segvalue

    @property
    def segmin(self):
        return self.__segMin

    @segmin.setter
    def segmin(self, segvalue):
        self.__segMin = segvalue

    @property
    def segsort(self):
        return self.__segSort

    @segsort.setter
    def segsort(self, sort_mode):
        self.__segSort = sort_mode

    @property
    def segstep(self):
        return self.__segStep

    @segstep.setter
    def segstep(self, segstep):
        self.__segStep = segstep

    @property
    def segalldata(self):
        return self.__usealldata

    @segalldata.setter
    def segalldata(self, datamode):
        self.__usealldata = datamode

    @property
    def display(self):
        return self.__display

    @display.setter
    def display(self, display):
        self.__display = display

    def set_data(self, raw_data, cols=None):
        self.__raw_dataframe = raw_data
        if type(cols) == str:
            cols = [cols]
        if (not isinstance(cols, list)) & isinstance(raw_data, pd.DataFrame):
            self.__cols = raw_data.columns.values
        else:
            self.__cols = cols
        self.__check()

    def set_para(
            self,
            segmax=None,
            segmin=None,
            segstart=None,
            segstep=None,
            seglist=None,
            segsort=None,
            useseglist=None,
            usealldata=None,
            display=None):
        set_str = ''
        if segmax is not None:
            self.__segMax = segmax
            set_str += 'set segmax to {}'.format(segmax) + '\n'
        if segmin is not None:
            self.__segMin = segmin
            set_str += 'set segmin to {}'.format(segmin) + '\n'
        if segstep is not None:
            self.__segStep = segstep
            set_str += 'set segstep to {}'.format(segstep) + '\n'
        if segstart is not None:
            set_str += 'set segstart to {}'.format(segstart) + '\n'
            self.__segStart = segstart
        if isinstance(segsort, str):
            if segsort.lower() in ['d', 'a', 'D', 'A']:
                set_str += 'set segsort to {}'.format(segsort) + '\n'
                self.__segSort = segsort
        if isinstance(usealldata, bool):
            set_str += 'set segalldata to {}'.format(usealldata) + '\n'
            self.__usealldata = usealldata
        if isinstance(display, bool):
            set_str += 'set display to {}'.format(display) + '\n'
            self.__display = display
        if isinstance(seglist, list):
            set_str += 'set seglist to {}'.format(seglist) + '\n'
            self.__segList = seglist
        if isinstance(useseglist, bool):
            set_str += 'set seglistuse to {}'.format(useseglist) + '\n'
            self.__useseglist = useseglist
        if display:
            print(set_str)
        self.__check()
        if display:
            self.show_para()

    def show_para(self):
        print('------ seg para ------')
        print('    use seglist:{}'.format(self.__useseglist))
        print('        seglist:{}'.format(self.__segList))
        print('       maxvalue:{}'.format(self.__segMax))
        print('       minvalue:{}'.format(self.__segMin))
        print('       segstart:{}'.format(self.__segStart))
        print('        segstep:{}'.format(self.__segStep))
        print('        segsort:{}'.format('d (descending)' if self.__segSort in ['d', 'D'] else 'a (ascending)'))
        print('     usealldata:{}'.format(self.__usealldata))
        print('        display:{}'.format(self.__display))
        print('-' * 28)

    def help_doc(self):
        print(self.__doc__)

    def __check(self):
        if isinstance(self.__raw_dataframe, pd.Series):
            self.__raw_dataframe = pd.DataFrame(self.__raw_dataframe)
        if not isinstance(self.__raw_dataframe, pd.DataFrame):
            print('error: raw score data is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('error: segmax({}) is not greater than segmin({})!'.format(self.__segMax, self.__segMin))
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('error: segstep({}) is too small or big!'.format(self.__segStep))
            return False
        if not isinstance(self.__cols, list):
            if isinstance(self.__cols, str):
                self.__cols = [self.__cols]
            else:
                print('error: segfields type({}) error.'.format(type(self.__cols)))
                return False

        for f in self.__cols:
            if f not in self.raw_data.columns:
                print("error: field('{}') is not in raw_data fields({})".
                      format(f, self.raw_data.columns.values))
                return False
        if not isinstance(self.__usealldata, bool):
            print('error: segalldata({}) is not bool type!'.format(self.__usealldata))
            return False
        return True

    def run(self):
        sttime = time.clock()
        if not self.__check():
            return
        # create output dataframe with segstep = 1
        if self.__display:
            print('---seg calculation start---')
        seglist = [x for x in range(int(self.__segMin), int(self.__segMax + 1))]
        if self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__out_dataframe = pd.DataFrame({'seg': seglist})
        outdf = self.__out_dataframe
        for f in self.__cols:
            # calculate preliminary group count
            tempdf = self.raw_data
            tempdf.loc[:, f] = tempdf[f].apply(round45r)

            # count seg_count in [segmin, segmax]
            r = tempdf.groupby(f)[f].count()
            fcount_list = [np.int64(r[x]) if x in r.index else 0 for x in seglist]

            outdf.loc[:, f+'_count'] = fcount_list
            if self.__display:
                print('finished count(' + f, ') use time:{}'.format(time.clock() - sttime))

            # add outside scope number to segmin, segmax
            if self.__usealldata:
                outdf.loc[outdf.seg == self.__segMin, f + '_count'] = \
                    r[r.index <= self.__segMin].sum()
                outdf.loc[outdf.seg == self.__segMax, f + '_count'] = \
                    r[r.index >= self.__segMax].sum()

            # calculate cumsum field
            outdf[f + '_sum'] = outdf[f + '_count'].cumsum()
            if self.__useseglist:
                outdf[f + '_list_sum'] = outdf[f + '_count'].cumsum()

            # calculate percent field
            maxsum = max(max(outdf[f + '_sum']), 1)     # avoid divided by 0 in percent computing
            outdf[f + '_percent'] = \
                outdf[f + '_sum'].apply(lambda x: round45r(x/maxsum, self.__percent_decimal))
            if self.__display:
                print('segments count finished[' + f, '], used time:{}'.format(time.clock() - sttime))

            # self.__out_dataframe = outdf.copy()
            # special seg step
            if self.__segStep > 1:
                self.__run_special_step(f)

            # use seglist
            if self.__useseglist:
                if len(self.__segList) > 0:
                    self.__run_seg_list(f)

        if self.__display:
            print('segments count total consumed time:{}'.format(time.clock()-sttime))
            print('---seg calculation end---')
        self.__run_completed = True
        self.__out_dataframe = outdf
        return

    def __run_special_step(self, field: str):
        """
        processing count for step > 1
        :param field: for seg stepx
        :return: field_countx in out_data
        """
        f = field
        segcountname = f + '_count{0}'.format(self.__segStep)
        self.__out_dataframe[segcountname] = np.int64(-1)
        curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        curpoint = self.__segStart
        if self.__segSort.lower() == 'd':
            while curpoint+curstep > self.__segMax:
                curpoint += curstep
        else:
            while curpoint+curstep < self.__segMin:
                curpoint += curstep
        # curpoint = self.__segStart
        cum = 0
        for index, row in self.__out_dataframe.iterrows():
            cum += row[f + '_count']
            curseg = np.int64(row['seg'])
            if curseg in [self.__segMax, self.__segMin]:
                self.__out_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if (self.__segStart <= self.__segMin) | (self.__segStart >= self.__segMax):
                    curpoint += curstep
                continue
            if curseg in [self.__segStart, curpoint]:
                self.__out_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                curpoint += curstep

    def __run_seg_list(self, field):
        """
        use special step list to create seg
        calculating based on field_count
        :param field:
        :return:
        """
        f = field
        segcountname = f + '_list'
        self.__out_dataframe[segcountname] = np.int64(-1)
        segpoint = sorted(self.__segList) \
            if self.__segSort.lower() == 'a' \
            else sorted(self.__segList)[::-1]
        # curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        # curpoint = self.__segStart
        cum = 0
        pos = 0
        curpoint = segpoint[pos]
        rownum = len(self.__out_dataframe)
        cur_row = 0
        lastindex = 0
        maxpoint = max(self.__segList)
        minpoint = min(self.__segList)
        list_sum = 0
        self.__out_dataframe.loc[:, f+'_list_sum'] = 0
        for index, row in self.__out_dataframe.iterrows():
            curseg = np.int64(row['seg'])
            # cumsum
            if self.__usealldata | (minpoint <= curseg <= maxpoint):
                cum += row[f + '_count']
                list_sum += row[f+'_count']
                self.__out_dataframe.loc[index, f+'_list_sum'] = np.int64(list_sum)
            # set to seg count, only set seg in seglist
            if curseg == curpoint:
                self.__out_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if pos < len(segpoint)-1:
                    pos += 1
                    curpoint = segpoint[pos]
                else:
                    lastindex = index
            elif cur_row == rownum:
                if self.__usealldata:
                    self.__out_dataframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1

    def plot(self):
        if not self.__run_completed:
            if self.__display:
                print('result is not created, please run!')
            return
        legendlist = []
        step = 0
        for sf in self.__cols:
            step += 1
            legendlist.append(sf)
            plot.figure('map_table figure({})'.
                       format('Descending' if self.__segSort in 'aA' else 'Ascending'))
            plot.subplot(221)
            plot.hist(self.raw_data[sf], 20)
            plot.title('histogram')
            if step == len(self.__cols):
                plot.legend(legendlist)
            plot.subplot(222)
            plot.plot(self.out_data.seg, self.out_data[sf+'_count'])
            if step == len(self.__cols):
                plot.legend(legendlist)
            plot.title('distribution')
            plot.xlim([self.__segMin, self.__segMax])
            plot.subplot(223)
            plot.plot(self.out_data.seg, self.out_data[sf + '_sum'])
            plot.title('cumsum')
            plot.xlim([self.__segMin, self.__segMax])
            if step == len(self.__cols):
                plot.legend(legendlist)
            plot.subplot(224)
            plot.plot(self.out_data.seg, self.out_data[sf + '_percent'])
            plot.title('percentage')
            plot.xlim([self.__segMin, self.__segMax])
            if step == len(self.__cols):
                plot.legend(legendlist)
            plot.show()
# SegTable class end


def round45i(v: float, dec=0):
    u = int(v * 10 ** dec * 10)
    r = (int(u / 10) + (1 if v > 0 else -1)) / 10 ** dec if (abs(u) % 10 >= 5) else int(u / 10) / 10 ** dec
    return int(r) if dec <= 0 else r


def round45r(number, digits=0):
    int_len = len(str(int(abs(number))))
    if int_len + abs(digits) <= 16:
        err_ = (1 if number >= 0 else -1)*10**-(16-int_len)
        if digits > 0:
            return round(number + err_, digits) + err_
        else:
            return int(round(number + err_, digits))
    else:
        raise NotImplemented


def round45r_old2(number, digits=0):
    __doc__ = '''
    float is not precise at digit 16 from decimal point.
    if hope that round(1.265, 3): 1.264999... to 1.265000...
    need to add a tiny error to 1.265: round(1.265 + x*10**-16, 3) => 1.265000...
    note that: 
        10**-16     => 0.0...00(53)1110011010...
        2*10**-16   => 0.0...0(52)1110011010...
        1.2*10**-16 => 0.0...0(52)100010100...
    so 10**-16 can not definitely represented in float 1+52bit

    (16 - int_len) is ok, 17 is unstable
    test result:
    format(1.18999999999999994671+10**-16, '.20f')     => '1.1899999999999999(16)4671'      ## digit-16 is reliable
    format(1.18999999999999994671+2*10**-16, '.20f')   => '1.1900000000000001(16)6875'
    format(1.18999999999999994671+1.2*10**-16, '.20f') => '1.1900000000000001(16)6875'
    format(1.18999999999999994671+1.1*10**-16, '.20f') => '1.1899999999999999(16)4671'
    '''

    int_len = str(abs(number)).find('.')
    if int_len + digits > 16:
        print('float cannot support {} digits precision'.format(digits))
        raise ValueError
    add_err = 10**-12       # valid for 0-16000
    # add_err = 3.55275*10**-15
    # add_err = 2*10**-14
    # add_err = 2 * 10 ** -(16 - int_len + 1) * (1 if number > 0 else -1)
    # if format(number, '.' + str(16 - digits - int_len) + 'f').rstrip('0') <= str(number):
    #     return round(number + add_err, digits) + add_err
    return round(number+add_err, digits)


def round45r_old(number, digits=0):
    __doc__ = '''
    use multiple 10 power and int method
    precision is not normal at decimal >16 because of binary representation
    :param number: input float value
    :param digits: places after decimal point
    :return: rounded number with assigned precision
    '''
    if format(number, '.'+str(digits+2)+'f').rstrip('0') <= str(number):
        return round(number+10**-(digits+2), digits)
    return round(number, digits)


def get_norm_dist_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000, decimal=6):
    """
    生成具有正态分布的数据，类型为 pandas.DataFrame, 列名为 sv
    create a score dataframe with fields 'score', used to test some application
    :parameter
        mean: 均值， std:标准差， maxvalue:最大值， minvalue:最小值， size:样本数
    :return
        DataFrame, columns = {'sv'}
    """
    # df = pd.DataFrame({'sv': [max(minvalue, min(int(np.random.randn(1)*std + mean), maxvalue))
    #                           for _ in range(size)]})
    df = pd.DataFrame({'sv': [max(minvalue,
                                  min(round45i(x, decimal) if decimal > 0 else int(round45i(x, decimal)),
                                      maxvalue))
                              for x in np.random.normal(mean, std, size)]})
    return df


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def get_norm_dist_table(size=400, std=1, mean=0, stdnum=4):
    """
    function
        生成正态分布量表
        create normal distributed data(pdf,cdf) with preset std,mean,samples size
        变量区间： [-stdNum * std, std * stdNum]
        interval: [-stdNum * std, std * stdNum]
    parameter
        变量取值数 size: variable value number for create normal distributed PDF and CDF
        分布标准差  std:  standard difference
        分布均值   mean: mean value
        标准差数 stdnum: used to define data range [-std*stdNum, std*stdNum]
    return
        DataFrame: 'sv':stochastic variable value,
                  'pdf': pdf value, 'cdf': cdf value
    """
    interval = [mean - std * stdnum, mean + std * stdnum]
    step = (2 * std * stdnum) / size
    varset = [mean + interval[0] + v*step for v in range(size+1)]
    cdflist = [sts.norm.cdf(v) for v in varset]
    pdflist = [sts.norm.pdf(v) for v in varset]
    ndf = pd.DataFrame({'sv': varset, 'cdf': cdflist, 'pdf': pdflist})
    return ndf
