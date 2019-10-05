# -*- utf-8 -*-


# comments to stm
"""
    2018.09.24 -- 2018.11
    2019.09.03 --
    designed for new High Test grade score model
    also for shandong interval linear transform

    stm module description stm模块说明：

    [functions] 模块中的函数
       run(name, df, field_list, ratio_list, grade_max, grade_diff, input_score_max, input_score_min,
           output_score_decimal=0, mode_ratio_approx='near', mode_ratio_cumu='yes')
          运行各个模型的调用函数 calling model function
          ---
          参数描述
          name:= 'shandong'/'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tao'
          调用山东、上海、浙江、北京、天津、广州、海南、...等模型进行分数转换
          caculate shandong... model by name = 'shandong' / 'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tao'
          -
          name:= 'zscore'/'tscore'/'tlinear'
          计算Z分数、T分数、线性转换T分数
          caculate Z,T,liear T score by name = 'zscore'/ 'tscore' / 'tlinear'
          --
          df: input raw score data, type DataFrame of pandas
          输入原始分数数据，类型为DataFrame
          --
          field_list: score field to calculate in df
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
          input_score_max: raw score max value
          最大原始分数
          --
          input_score_min: raw score min value
          最小原始分数
          --
          output_score_decimal: grade score precision, decimal digit number
          输出分数精度，小数位数
          --
          mode_ratio_approx: how to approxmate score points of raw score for each ratio vlaue
          计算等级时的逼近方式（目前设计的比例值逼近策略)：
              'upper_min': get score with min value in bigger 小于该比例值的分值中最大的值
              'lower_max': get score with max value in less 大于该比例值的分值中最小的值
              'near':   get score with nearest ratio 最接近该比例值的分值（分值）
              'near_min': get score with min value in near 最接近该比例值的分值中最小的值
              'near_max': get score with max value in near 最接近该比例值的分值中最大的值
              注1：针对等级划分区间，也可以考虑使用ROUND_HALF_UP，即靠近最近，等距时向上靠近
              注2：搜索顺序分为Big2Small和Small2Big两类，区间位精确的定点小数，只有重合点需要策略（UP或DOWN）

              拟改进为（2019.09.09） mode_ratio_approx：
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
          [2] m = stm.run(name='shandong', df=data, field_list=['ls'])
          [3] m.report()
          [4] m.output.head()
          [5] m.save_output_data_to_csv

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
       TaoScore: 陶百强等级分数模型（由陶百强在其论文中提出）Tao Baiqiang model
       Zscore: Z分数转换模型 zscore model
       Tscore: T分数转换模型 tscore model
       Tlinear: T分数线性转换模型 tscore model by linear transform mode
       SegTable: 计算分段表模型 segment table model

    [CONSTANTS] 模块中的常量
        各省市等级分数转换比例设置，山东省区间划分设置
        CONST_ZHEJIANG_RATIO = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        CONST_SHANGHAI_RATIO = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
        CONST_BEIJING_RATIO = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
        CONST_TIANJIN_RATIO = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]
        CONST_SHANDONG_RATIO = [3, 7, 16, 24, 24, 16, 7, 3]
        CONST_SHANDONG_SEGMENT = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
        CONST_M7_RATIO = [2, 13, 35, 35, 15]
        CONST_M7_SEGMENT = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]
"""


# built-in import
import copy
import time
import os
import warnings
import fractions as fr
from collections import namedtuple
# import decimal as dc


# external import
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
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
#            mean = sum([x/100*sum(y)/2 for x,y in zip(M7ratio,M7segment)])
#            std = math.sqrt(sum([(sum(y)/2-mean)**2 for x,y in zip(M7ratio,M7segment)])/5)
CONST_M7_RATIO = [15, 35, 35, 13, 2]
CONST_M7_SEGMENT = [(100, 86), (85, 71), (70, 56), (55, 41), (40, 30)]

Plt = namedtuple('Plt', ['ratio', 'seg'])
plt_models_dict = {
    'zhejiang': Plt(CONST_ZHEJIANG_RATIO, CONST_ZHEJIANG_SEGMENT),
    'shanghai': Plt(CONST_SHANGHAI_RATIO, CONST_SHANGHAI_SEGMENT),
    'beijing': Plt(CONST_BEIJING_RATIO, CONST_BEIJING_SEGMENT),
    'tianjin': Plt(CONST_TIANJIN_RATIO, CONST_TIANJIN_SEGMENT),
    'shandong': Plt(CONST_SHANDONG_RATIO, CONST_SHANDONG_SEGMENT),
    'guangdong': Plt(CONST_GUANGDONG_RATIO, CONST_GUANGDONG_SEGMENT),
    'm7': Plt(CONST_M7_RATIO, CONST_M7_SEGMENT)
    }
plt_models_strategies_dict ={
    'mode_score_order': ['a', 'ascending', 'd', 'descending'],
    'mode_ratio_approx': ['upper_min', 'lower_max', 'near_max', 'near_min'],
    'mode_ratio_cumu': ['yes', 'y', 'no', 'n'],
    'mode_endpoint_share': ['yes', 'y', 'no', 'n'],
    'mode_seg_degraded': ['max', 'min', 'mean'],
    'mode_score_zero': ['use', 'ignore'],
    'mode_score_empty': ['use', 'ignore'],
    'mode_score_max': ['real', 'full_mark'],
    'mode_score_min': ['real', 'zero']
    }
stm_models_name = ['z', 't', 'hainan', 'tao'] + \
                  list(plt_models_dict.keys())


def about_stm():
    print(__doc__)


def test_stm(
        model='shandong',
        max_score=100,
        min_score=0,
        data_size=1000):

    if model.lower() not in stm_models_name:
        print('correct model name in: [{}]'.format(','.join(stm_models_name)))
        return None

    # create data set
    print('create test dataset...')
    # dfscore = pd.DataFrame({'km': np.random.randint(0, max_score, data_size, 'int')})
    norm_data = [sts.norm.rvs() for _ in range(data_size)]
    norm_data = [-4 if x < -4 else (4 if x > 4 else x) for x in norm_data]
    norm_data = [int(x*(max_score-min_score)/8 + (max_score+min_score)/2) for x in norm_data]
    dfscore = pd.DataFrame({'kmx': norm_data})

    if model in stm_models_name[2:]:
        print('test model={}'.format(model))
        print('data set size={}, score range from 0 to 100'.format(data_size))
        m = run_stm(name=model, df=dfscore, field_list='kmx')
        return m
    elif model.lower() == 'z':
        m = Zscore()
        m.set_data(dfscore, field_list=['km'])
        m.set_para(rawscore_max=max_score, rawscore_min=min_score)
        m.run()
        return m
    elif model.lower() == 't':
        m = Tscore()
        m.set_data(dfscore, field_list=['km'])
        m.set_para(rawscore_max=100, rawscore_min=0,
                   tscore_mean=500, tscore_std=100, tscore_stdnum=4)
        m.run()
        return m
    return None


# interface to use model for some typical application
def run_stm(
        name='shandong',
        df=None,
        field_list='',
        input_score_max=None,
        input_score_min=None,
        output_score_decimal=0,
        mode_ratio_approx='upper_min',
        mode_ratio_cumu='yes',
        mode_score_order='descending'
        ):
    """
    :param name: str, 'shandong', 'shanghai', 'shandong', 'beijing', 'tianjin', 'zscore', 'tscore', 'tlinear'
                      'guangdong', 'M7', default = 'shandong'
    :param df: dataframe, input data, default = None
    :param field_list: score fields list in input dataframe, default = None and set to digit fields in running
    :param ratio_list: ratio list used to create intervals of raw score for each grade
                        default = None, set to a list by the model's name
                        must be set to a list if name is not in module preassigned list
                        must be set for new model
    :param grade_diff: difference value between two neighbor grade score
                        default = None, that will be set to 3 if name in 'zhejiang, shanghai, beijing, tianjin'
                        must be set for new model
    :param grade_max: max value for grade score
                       default = None, will be set to 100 for zhejiang,shanghai,beijing,tianjin, shandong
                       must be set for new model
    :param input_score_max: max value in raw score
                       default = None, set to 150 in ScoreTransform, set to real max value in PltScore, GradeScore
    :param input_score_min: min value in raw score
                       default = None, set to 0 in ScoreTransform, set to real min value in PltScore, GradeScore
    :param output_score_decimal: output score decimal digits
                                  default = 0 for int score at output score
    :param mode_ratio_approx: lower_max, upper_min, near(near_max, near_min)  default=lower_max  # for shandong new project
    :param mode_ratio_cumu: yes, no  default=yes                     # for shandong new project
    :param mode_score_order: descending(from max to min), ascending(from min to max)
    :return: model
    """
    # check name
    name = name.lower()
    if name.lower() not in stm_models_name:
        print('invalid name, not in {}'.format(stm_models_name))
        return

    # check input data
    if type(df) != pd.DataFrame:
        if type(df) == pd.Series:
            input_data = pd.DataFrame(df)
        else:
            print('no score dataframe!')
            return
    else:
        input_data = df

    # check field_list
    if isinstance(field_list, str):
        field_list = [field_list]
    elif not isinstance(field_list, list):
        print('invalid field_list!')
        return

    # check mode_ratio_approx
    if mode_ratio_approx not in ['lower_max', 'upper_min', 'near_min', 'near_max']:
        print('invalid approx mode: {}'.format(mode_ratio_approx))
        print('  valid approx mode: lower_max, upper_min, near_min, near_max')
        return
    if mode_ratio_cumu not in ['yes', 'no']:
        print('invalid cumu mode(yes/no): {}'.format(mode_ratio_cumu))
        return

    # plt score models
    if name in plt_models_dict.keys():        # ['shandong', 'guangdong', 'm7',
                                                      # 'zhejiang', 'shanghai', 'beijing', 'tianjin']:
        ratio_list = [x*0.01 for x in plt_models_dict[name].ratio]
        pltmodel = PltScore()
        pltmodel.model_name = name
        pltmodel.output_data_decimal = 0
        pltmodel.set_data(input_data=input_data,
                          field_list=field_list)
        pltmodel.set_para(input_score_ratio_list=ratio_list,
                          output_score_points_list=plt_models_dict[name].seg,
                          input_score_max=input_score_max,
                          input_score_min=input_score_min,
                          mode_ratio_approx=mode_ratio_approx,
                          mode_ratio_cumu=mode_ratio_cumu,
                          mode_score_order=mode_score_order,
                          decimals=output_score_decimal)
        pltmodel.run()
        return pltmodel

    if name == 'tao':
        m = GradeScoreTao()
        m.grade_num = 50
        m.set_data(input_data=input_data,
                   field_list=field_list)
        m.set_para(maxscore=input_score_max,
                   minscore=input_score_min)
        m.run()
        return m

    if name == 'zscore':
        zm = Zscore()
        zm.model_name = name
        zm.set_data(input_data=input_data, field_list=field_list)
        zm.set_para(std_num=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm

    if name == 'tscore':
        tm = Tscore()
        tm.model_name = name
        tm.set_data(input_data=input_data, field_list=field_list)
        tm.set_para(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm

    if name == 'tlinear':
        tm = TscoreLinear()
        tm.model_name = name
        tm.set_data(input_data=input_data, field_list=field_list)
        tm.set_para(input_score_max=input_score_max,
                    input_score_min=input_score_min)
        tm.run()
        tm.report()
        return tm


def plot_stm():
    ms_dict = stm_mean_std()
    pyplt.figure('Noew Gaokao Grade Score Distribution of Models')
    pyplt.rcParams.update({'font.size': 16})

    pyplt.subplot(241)
    pyplt.bar(range(40, 71, 3), CONST_SHANGHAI_RATIO[::-1])
    pyplt.title('Shanghai({:.2f}, {:.2f})'.format(*ms_dict['shanghai']))

    pyplt.subplot(242)
    pyplt.bar(range(40, 101, 3), CONST_ZHEJIANG_RATIO[::-1])
    pyplt.title('Zhejiang({:.2f}, {:.2f})'.format(*ms_dict['zhejiang']))

    pyplt.subplot(243)
    pyplt.bar(range(40, 101, 3), CONST_BEIJING_RATIO[::-1])
    pyplt.title('Beijing({:.2f}, {:.2f})'.format(*ms_dict['beijing']))

    pyplt.subplot(244)
    pyplt.bar(range(40, 101, 3), CONST_TIANJIN_RATIO[::-1])
    pyplt.title('Tianjin({:.2f}, {:.2f})'.format(*ms_dict['tianjin']))

    pyplt.subplot(245)
    sbn.barplot([x for x in range(25, 101, 10)], [CONST_SHANDONG_RATIO[j] for j in range(8)])
    pyplt.title('Shandong:({:.2f}, {:.2f})'.format(*ms_dict['shandong']))

    pyplt.subplot(246)
    sbn.barplot([np.mean(x) for x in CONST_GUANGDONG_SEGMENT][::-1], CONST_GUANGDONG_RATIO[::-1])
    pyplt.title('Guangdong({:.2f}, std={:.2f})'.format(*ms_dict['guangdong']))

    pyplt.subplot(247)
    sbn.barplot([np.mean(x) for x in CONST_M7_SEGMENT][::-1], CONST_M7_RATIO[::-1])
    pyplt.title('Jiangsu..({:.2f}, std={:.2f})'.format(*ms_dict['m7']))


def stm_mean_std():
    name_list = ['shandong', 'shanghai', 'zhejiang', 'guangdong', 'm7', 'beijing', 'tianjin']
    mean_std_dict = dict()
    for _name in name_list:
        mean_std_dict.update({_name: calc_stm_mean_std(name=_name)})
        # score_max = 100
        # score_gap = 3
        # ratio_lst = None
        # score_seg = None
        # if _name in ['shandong', 'guangdong', 'm7']:
        #     if _name =='shandong':
        #         ratio_lst = CONST_SHANDONG_RATIO
        #         score_seg = CONST_SHANDONG_SEGMENT
        #     if _name =='guangdong':
        #         ratio_lst = CONST_GUANGDONG_RATIO
        #         score_seg = CONST_GUANGDONG_SEGMENT
        #     if _name =='m7':
        #         ratio_lst = CONST_M7_RATIO
        #         score_seg = CONST_M7_SEGMENT
        #     mean_std_dict.update({_name: calc_stm_mean_std(name=_name)})
        # if _name in ['shanghai', 'zhejiang', 'beijing', 'tianjin']:
        #     if _name =='shanghai':
        #         ratio_lst = CONST_SHANGHAI_RATIO
        #         score_max = 70
        #     if _name =='zhejiang':
        #         ratio_lst = CONST_ZHEJIANG_RATIO
        #     if _name =='beijing':
        #         ratio_lst = CONST_BEIJING_RATIO
        #     if _name =='tianjin':
        #         ratio_lst = CONST_TIANJIN_RATIO
        #     mean_std_dict.update({_name: calc_stm_mean_std(name=_name)})
    return mean_std_dict


def calc_stm_mean_std(name='shandong'):
    _mean, _std = -1, -1
    _mean = sum([r / 100 * sum(s) / 2 for r, s in zip(plt_models_dict[name].ratio, plt_models_dict[name].seg)])
    _std = np.sqrt(sum([(sum(s)/2-_mean) ** 2 * plt_models_dict[name].ratio[i]
                        for i, s in enumerate(plt_models_dict[name].seg)]) / 100)
    return _mean, _std


def mentcaro(ratio=tuple(CONST_SHANDONG_RATIO),
             seg=tuple(CONST_SHANDONG_SEGMENT),
             size=1000,
             ):
    _loc=[x/100*sum(y)/2 for x, y in zip(ratio, seg)]
    _max = max([max(x) for x in seg])
    _min = min([min(x) for x in seg])
    _mu = (_max + _min)/2
    _sigma = (_max - _min + 1)/6
    raw_mean = 50
    raw_std = (100-raw_mean)/3
    ndata = np.random.normal(raw_mean, raw_std, size=size)
    ndata = [x if 0 <= x <= 100 else (0 if x < 0 else 100) for x in ndata]
    print('mu={:.2f}, sigma={:.2f} std={:.2f}, mean={:.2f}'.
          format(_mu, _sigma, np.std(ndata), np.mean(ndata)))
    m = run_stm(df=pd.DataFrame({'fs':ndata}), field_list='fs')
    return m


def run_seg(
            input_data:pd.DataFrame,
            field_list:list,
            segmax=100,
            segmin=0,
            segsort='d',
            segstep= 1,
            display=False,
            usealldata=False
            ):
    seg = SegTable()
    seg.set_data(
        input_data=input_data,
        field_list=field_list
    )
    seg.set_para(
        segmax=segmax,
        segmin=segmin,
        segstep=segstep,
        display=display,
        useseglist=usealldata
    )
    seg.run()
    return seg


# test dataset
class test_data():
    def __init__(self, mean=60, max=100, min=0, std=18, size=1000000):
        self.data_mean = mean
        self.data_max = max
        self.data_min = min
        self.data_std = std
        self.data_size = size
        self.data_set = None

    def get_data(self):
        self.data_set = pd.DataFrame({
            'no': [str(x).zfill(7) for x in range(1, self.data_size+1)],
            'km1': self.get_score(),
            'km2': self.get_score(),
            'km3': self.get_score(),
        })

    def get_score(self):
        print('create score...')
        score_list = sts.norm.rvs(loc=self.data_mean, scale=self.data_std, size=self.data_size)
        score_list = [(int(x) if x <= 100 else 100) if x >= 0 else 0
                      for x in score_list]
        return score_list


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
        param input_data: raw score data, type==datafrmae
        param field_list: fields in input_data, assign somr subjects score to transform
        param output_data: transform score data, type==dataframe
    """
    def __init__(self, model_name=''):
        self.model_name = model_name

        self.input_data = pd.DataFrame()
        self.field_list = []
        self.input_score_min = 0
        self.input_score_max = 150

        self.output_data = pd.DataFrame()
        self.output_data_decimal = 0
        self.output_report_doc = ''
        self.map_table = pd.DataFrame()

        self.sys_pricision_decimals = 8

    def set_data(self, input_data=None, field_list=None):
        raise NotImplementedError()

    def set_para(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if not isinstance(self.input_data, pd.DataFrame):
            print('rawdf is not dataframe!')
            return False
        if (type(self.field_list) != list) | (len(self.field_list) == 0):
            print('no score fields assigned!')
            return False
        for sf in self.field_list:
            if sf not in self.input_data.columns:
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

    def read_input_data_from_csv(self, filename=''):
        if not os.path.isfile(filename):
            print('{} not valid file name'.format(filename))
            return
        self.input_data = pd.read_csv(filename)

    def save_output_data_to_csv(self, filename):
        self.output_data.to_csv(filename, index=False)

    def save_report_to_file(self, filename):
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(self.output_report_doc)

    def save_map_table_to_file(self,filename):
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
        if not self.field_list:
            print('no field:{0} assign in {1}!'.format(self.field_list, self.input_data))
            return
        # pyplt.figure(self.model_name + ' out score figure')
        labelstr = 'Output Score '
        for fs in self.field_list:
            pyplt.figure(fs)
            if fs + '_plt' in self.output_data.columns:  # find sf_outscore field
                sbn.distplot(self.output_data[fs + '_plt'])
                pyplt.title(labelstr + fs)
            elif fs + '_grade' in self.output_data.columns:  # find sf_outscore field
                sbn.distplot(self.output_data[fs + '_grade'])
                pyplt.title(labelstr + fs)
            else:
                print('mode=out only for plt and grade model!')
        return

    def __plot_raw_score(self):
        if not self.field_list:
            print('no field assign in rawdf!')
            return
        labelstr = 'Raw Score '
        for sf in self.field_list:
            pyplt.figure(sf)
            sbn.distplot(self.input_data[sf])
            pyplt.title(labelstr + sf)
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
    # m7 Model Analysis
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
        # intit input_df, input_output_data, output_df, model_name
        super(PltScore, self).__init__('plt')

        # new properties for shandong model
        self.input_score_ratio_cum = []
        self.output_score_points = []
        self.output_data_decimal = 0
        self.output_score_max = None
        self.output_score_min = None

        # para
        self.mode_ratio_approx = 'upper_min'
        self.mode_ratio_cumu = 'yes'
        self.mode_score_order = 'descending'
        self.mode_seg_degraded = 'max'
        self.mode_score_zero = 'use'
        self.mode_score_max = 'real'
        self.mode_score_min = 'real'
        self.mode_score_empty = 'ignore'
        self.mode_endpoint_share = 'no'
        # self.use_min_rawscore_as_endpoint = True
        # self.use_max_rawscore_as_endpoint = True

        # result
        self.seg_model = None
        self.map_table = pd.DataFrame()
        self.result_input_data_points = []
        self.result_ratio_dict = {}
        self.result_formula_coeff = {}
        self.result_formula_text_list = ''
        self.result_dict = {}

    def set_data(self, input_data=None, field_list=None):

        # check and set rawdf
        if type(input_data) == pd.Series:
            self.input_data = pd.DataFrame(input_data)
        elif type(input_data) == pd.DataFrame:
            self.input_data = input_data
        else:
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set output_data
        if not field_list:
            self.field_list = [s for s in input_data]
        elif type(field_list) != list:
            print('field_list set fail!\n not a list!')
            return
        elif sum([1 if sf in input_data else 0 for sf in field_list]) != len(field_list):
            print('field_list set fail!\n field must in rawdf.columns!')
            return
        else:
            self.field_list = field_list

    def set_para(self,
                 input_score_ratio_list=None,
                 output_score_points_list=None,
                 input_score_min=None,
                 input_score_max=None,
                 mode_ratio_approx='upper_min',
                 mode_ratio_cumu='yes',
                 mode_score_order='descending',
                 decimals=None):
        """
        :param input_score_ratio_list: ratio points for raw score interval
        :param output_score_points_list: score points for output score interval
        :param input_score_min: min value to transform
        :param input_score_max: max value to transform
        :param mode_ratio_approx:  upper_min, lower_max, near_min, near_max
        :param mode_score_order: search ratio points from high score to low score if 'descending' or
                            low to high if 'descending'
        :param decimals: decimal digit number to remain in output score
        """
        if (type(input_score_ratio_list) != list) | (type(output_score_points_list) != list):
            print('input score points or output score points is not list!')
            return
        if len(input_score_ratio_list) != len(output_score_points_list):
            print('the number of input score points is not same as output score points!')
            return
        if mode_ratio_cumu not in 'yes, no':
            print('mode_ratio_cumu value error:{}'.format(mode_ratio_cumu))

        if isinstance(decimals, int):
            self.output_data_decimal = decimals

        input_p = None
        if mode_score_order in ['descending', 'd']:
            input_p = input_score_ratio_list
            out_pt = output_score_points_list
        else:
            input_p = input_score_ratio_list[::-1]
            out_pt = output_score_points_list[::-1]
        self.output_score_points = [x[::-1] for x in out_pt]
        self.output_score_points = output_score_points_list
        self.input_score_ratio_cum = [sum(input_p[0:x + 1]) for x in range(len(input_p))]

        self.input_score_min = input_score_min
        self.input_score_max = input_score_max

        self.mode_ratio_approx = mode_ratio_approx
        self.mode_ratio_cumu = mode_ratio_cumu
        self.mode_score_order = mode_score_order

    def check_parameter(self):
        if not self.field_list:
            print('no score field assign in field_list!')
            return False
        if (type(self.input_score_ratio_cum) != list) | (type(self.output_score_points) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return False
        if (len(self.input_score_ratio_cum) != len(self.output_score_points)) | \
                len(self.input_score_ratio_cum) == 0:
            print('len is 0 or not same for raw score percent and std score points list!')
            return False
        return True
    # --------------data and para setting end

    def run(self):

        print('stm-run begin...\n'+'-'*50)
        stime = time.time()

        # check valid
        if not super(PltScore, self).run():
            return

        if self.input_score_max is None:
            # self.input_score_max = max([self.input_data[fs].max() for fs in self.field_list])
            self.input_score_max = max(self.input_data[self.field_list].max())
        if self.input_score_min is None:
            # self.input_score_min = min([self.input_data[fs].min() for fs in self.field_list])
            self.input_score_min = min(self.input_data[self.field_list].min())
        if self.output_score_points is not None:
            self.output_score_max = max([max(x) for x in self.output_score_points])
            self.output_score_min = min([min(x) for x in self.output_score_points])

        # calculate seg table
        print('--- calculating map_table ...')
        self.seg_model = run_seg(
                  input_data=self.input_data,
                  field_list=self.field_list,
                  segmax=self.input_score_max,
                  segmin=self.input_score_min,
                  segsort='a' if self.mode_score_order in ['ascending', 'a'] else 'd',
                  segstep=1,
                  display=False,
                  usealldata=False
                  )
        self.map_table = self.seg_model.output_data   # .copy(deep=True)

        # create field_fr in map_table
        #   strange error!!: some seg percent to zero
        #   self.map_table[f+'_percent'] = self.map_table[f+'_fr'].apply(lambda x: float(x))
        for f in self.field_list:
            max_sum = max(self.map_table[f+'_sum'])
            max_sum = 1 if max_sum == 0 else max_sum
            self.map_table[f+'_fr'] = self.map_table[f+'_sum'].\
                                      apply(lambda x: fr.Fraction(x, max_sum))
            self.map_table.astype({f+'_fr': fr.Fraction})

        # transform score on each field
        self.output_report_doc = 'Transform Model: [{}]\n'.format(self.model_name)
        self.output_report_doc += '---'*40 + '\n'

        # algorithm strategy
        self.output_report_doc += format('strategies: ', '>23') + '\n'

        self.output_report_doc += ' '*23 + 'score_order = {},\t\t\t\t {}'.\
            format(self.mode_score_order, plt_models_strategies_dict['mode_score_order']) + '\n'
        self.output_report_doc += ' '*23 + 'ratio_approx = {},\t {}'.\
            format(self.mode_ratio_approx, plt_models_strategies_dict['mode_ratio_approx']) + '\n'
        self.output_report_doc += ' '*23 + 'ratio_cumu = {},\t\t\t {}'.\
            format(self.mode_ratio_cumu, plt_models_strategies_dict['mode_ratio_cumu']) + '\n'
        self.output_report_doc += ' '*23 + 'seg_degraded = {},\t\t\t {}'.\
            format(self.mode_seg_degraded, plt_models_strategies_dict['mode_seg_degraded']) + '\n'
        self.output_report_doc += ' '*23 + 'score_zero = {},\t\t\t {}'.\
            format(self.mode_score_zero, plt_models_strategies_dict['mode_score_zero']) + '\n'
        self.output_report_doc += ' '*23 + 'score_max = {},\t\t\t {}'.\
            format(self.mode_score_max, plt_models_strategies_dict['mode_score_max']) + '\n'
        self.output_report_doc += ' '*23 + 'score_min = {},\t\t\t {}'.\
            format(self.mode_score_min, plt_models_strategies_dict['mode_score_min']) + '\n'
        self.output_report_doc += ' '*23 + 'score_empty = {},\t\t {}'.\
            format(self.mode_score_empty, plt_models_strategies_dict['mode_score_empty']) + '\n'
        self.output_report_doc += ' '*23 + 'endpoints_share = {},\t\t {}'.\
            format(self.mode_endpoint_share, plt_models_strategies_dict['mode_endpoint_share']) + '\n'

        # self.output_report_doc += '- -'*40 + '\n'
        # for k in plt_models_strategies_dict.keys():
        #     self.output_report_doc += ' ' * 23 + '{} = {}'.\
        #         format(k, plt_models_strategies_dict[k]) + '\n'
        self.output_report_doc += '---'*40 + '\n'

        self.result_dict = dict()
        self.output_data = self.input_data.copy(deep=True)
        for i, fs in enumerate(self.field_list):
            print('--- transform score field:[{}]'.format(fs))

            # get formula
            if not self.__get_formula(fs):
                print('fail to get formula !')
                return

            # save result_formula, seg
            self.result_dict[fs] = {
                'input_score_points': copy.deepcopy(self.result_input_data_points),
                'coeff': copy.deepcopy(self.result_formula_coeff),
                'formulas': copy.deepcopy(self.result_formula_text_list)}

            # get field_plt in output_data
            print('   calculate: {0} => {0}_plt'.format(fs))
            self.output_data.loc[:, (fs + '_plt')] = \
                self.output_data[fs].apply(
                    lambda x: self.get_plt_score_from_formula3(fs, x))

            if self.output_data_decimal == 0:
                self.output_data[fs] = self.output_data[fs].astype('int')
                self.output_data[fs+'_plt'] = self.output_data[fs+'_plt'].astype('int')

            print('   create report ...')
            self.output_report_doc += self.__get_report_doc(fs)

        # get fs_plt in map_table
        df_map = self.map_table
        for fs in self.field_list:
            fs_name = fs + '_plt'
            df_map.loc[:, fs_name] = df_map['seg'].apply(
                lambda x: self.get_plt_score_from_formula3(fs, x))
            if self.output_data_decimal == 0:
                df_map[fs_name] = df_map[fs_name].astype('int')

        print('-'*50)
        print('stm-run end, elapsed-time:', time.time() - stime)

    # run end

    # -----------------------------------------------------------------------------------
    # formula-1
    # y = a*x + b
    def get_plt_score_from_formula1(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                return round45r(cf[0][0] * x + cf[0][1])
        return -1

    # -----------------------------------------------------------------------------------
    # formula-2
    # y = a*(x - b) + c
    def get_plt_score_from_formula2(self, field, x):
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
    # formula-3 new, recommend to use
    # y = (a*x + b) / c : a=(y2-y1), b=y1x2-y2x1, c=(x2-x1)
    def get_plt_score_from_formula3(self, field, x):
        if x >= self.input_score_max:
            return self.output_score_max
        if x <= self.input_score_min:
            return self.output_score_min
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                a = (cf[2][1]-cf[2][0])
                b = cf[2][0]*cf[1][1] - cf[2][1]*cf[1][0]
                c = (cf[1][1]-cf[1][0])
                # x1 == x2 then return max(y1, y2)
                if c == 0:
                    return max(cf[2])
                return round45r((a*x + b)/c)
        return -1

    def __get_formula(self, field):
        # --step 1
        # claculate rawscore_endpoints
        if field in self.output_data.columns.values:
            print('   get input score endpoints ...')
            # points_list = self.__get_raw_score_from_ratio(field, self.mode_ratio_approx)
            points_list = self.__get_formula_raw_seg_list(field=field,
                                                          mode_ratio_approx=self.mode_ratio_approx,
                                                          cum_mode=self.mode_ratio_cumu,
                                                          mode_score_order=self.mode_score_order,
                                                          score_max=self.input_score_max,
                                                          score_min=self.input_score_min,
                                                          raw_score_ratio_cum_list=self.input_score_ratio_cum,
                                                          map_table=self.map_table)
            self.result_input_data_points = points_list
            if len(points_list) == 0:
                return False
        else:
            print('score field({}) not in output_dataframe columns:{}!'.format(field, self.output_data.columns.values))
            print('the field should be in input_dataframe columns:{}'.format(self.input_data.columns.values))
            return False

        # --step 2
        # calculate Coefficients
        if not self.__get_formula_coeff():
            return False
        return True

    # -----------------------------------------------------------------------------------
    # formula-1: y = (y2-y1)/(x2 -x1)*(x - x1) + y1                   # a(x - b) + c
    #        -2:   = (y2-y1)/(x2 -x1)*x + (y1x2 - y2x1)/(x2 - x1)     # ax + b
    #        -3:   = [(y2-y1)*x + y1x2 - y2x1]/(x2 - x1)              # (ax + b) / c ; int / int
    def __get_formula_coeff(self):

        # calculate coefficient
        x_points = self.result_input_data_points
        step = 1 if self.mode_score_order in ['ascending', 'a'] else -1
        x_list = [(int(p[0])+(step if i > 0 else 0), int(p[1]))
              for i, p in enumerate(zip(x_points[:-1], x_points[1:]))]
        y_list = self.output_score_points
        for i, endpointxy in enumerate(zip(x_list, y_list)):
            x, y = endpointxy
            v = x[1] - x[0]
            if v == 0:
                a, b = 0, max(y)                    # x1 == x2 : y = max(y1, y2)
            else:
                a = (y[1]-y[0])/v                   # (y2 - y1) / (x2 - x1)
                b = (y[0]*x[1]-y[1]*x[0])/v         # (y1x2 - y2x1) / (x2 - x1)
            self.result_formula_coeff.update({i: [(a, b), x, y]})
        return True

    # new at 2019-09-09
    def __get_formula_raw_seg_list(self,
                                   field,
                                   mode_ratio_approx='lower_max',
                                   cum_mode='yes',
                                   mode_score_order='d',  # from high to low
                                   score_min=0,
                                   score_max=100,
                                   raw_score_ratio_cum_list=None,
                                   map_table=None):
        result_ratio = []
        # start and end points for raw score segments
        raw_score_start = score_min if mode_score_order in ['a', 'ascending'] \
                          else self.input_data[field].max()     # score_max
        raw_score_end = score_max if mode_score_order in ['a', 'ascending'] \
                          else self.input_data[field].min()     # score_min
        result_raw_seg_list = [raw_score_start]
        last_ratio = 0
        last_percent = 0
        for i, ratio in enumerate(raw_score_ratio_cum_list):
            dest_percent = ratio if cum_mode == 'no' else ratio-last_ratio+last_percent
            if i == len(raw_score_ratio_cum_list)-1:
                dest_percent = 1.0
            result_this_seg_endpoint, result_this_seg_percent = \
                self.get_seg_from_map_table(
                                            map_table=map_table,
                                            field=field,
                                            dest_ratio=dest_percent,
                                            mode_ratio_approx=mode_ratio_approx)
            last_ratio = ratio
            last_percent = result_this_seg_percent
            if ratio == raw_score_ratio_cum_list[-1]:
                result_this_seg_endpoint = raw_score_end
            if result_this_seg_endpoint < 0:
                result_raw_seg_list.append(raw_score_end)
            else:
                result_raw_seg_list.append(result_this_seg_endpoint)
            result_ratio.append('{:.4f}'.format(result_this_seg_percent))
            # if result_raw_seg_list[-1] >= 0:
            print('   <{}> ratio: [def:{:.2f} dest:{:.4f} result:{:.4f}] => raw_seg: [{:3.0f}, {:3.0f}]'.
                  format(i+1, ratio, dest_percent, result_this_seg_percent,
                         result_raw_seg_list[-2] if i==0 else result_raw_seg_list[-2]-1,
                         result_this_seg_endpoint))
            # else:
            #     print('   <{}> {}'.format(i+1, '******'))

        self.result_ratio_dict[field] = result_ratio
        return result_raw_seg_list

    # new at 2019-09-09
    def get_seg_from_map_table(self,
                               map_table,
                               field,
                               dest_ratio,
                               mode_ratio_approx):

        _mode = mode_ratio_approx.lower().strip()
        result_seg_endpoint = min(map_table[field+'_percent']) \
            if self.mode_score_order in ['descending', 'd'] \
            else max(map_table[field])
        _seg = -1
        _percent = -1
        last_percent = -1
        last_seg = None
        last_diff = 1000
        _use_last = None
        for index, row in map_table.iterrows():
            _percent = row[field+'_percent']
            _seg = row['seg']
            _diff = abs(_percent - dest_ratio)

            # at bottom
            if index == map_table.index.max():
                _use_last = False

            # reach bigger than or equal to ratio
            if _percent >= dest_ratio:
                # at top
                if last_seg is None:
                    _use_last = False

                # dealing with tragedies
                if 'near' in _mode:
                    # this ratio is near
                    if _diff < last_diff:
                        _use_last = False
                    # last ratio is near
                    elif _diff > last_diff:
                        _use_last = True
                    # distances are same
                    else:
                        # mode is near_min
                        if 'near_min' in _mode:
                            _use_last = True
                        # mode is near_max
                        else:
                            _use_last = False
                elif _mode == 'lower_max':
                    if _percent == dest_ratio:
                        _use_last = False
                    else:
                        _use_last = True
                elif _mode == 'upper_min':
                    _use_last = False
            if _use_last is not None:
                break
            last_seg = _seg
            last_diff = _diff
            last_percent = _percent
        if _use_last:
            return last_seg, last_percent
        else:
            return _seg, _percent

    def get_seg_from_fr(self, mapdf, field, ratio):
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


    def __get_report_doc(self, field=''):
        p = 0 if self.mode_score_order in ['ascending', 'a'] else 1
        self.result_formula_text_list = []
        for k in self.result_formula_coeff:
            formula = self.result_formula_coeff[k]
            if formula[0][0] > 0:
                self.result_formula_text_list += \
                    ['(seg-{0}) y = {1:0.6f}*(x-{2:2d}) + {3:2d}'.
                         format(k+1, formula[0][0], formula[1][p], formula[2][p])]
            elif formula[0][0] == 0:
                self.result_formula_text_list += \
                    ['(seg-{0}) y = {1:0.6f}*(x-{2:2d}) + max({3:2d}, {4:2d})'.
                         format(k + 1, formula[0][0], formula[1][p], formula[2][0], formula[2][1])]
            else:
                self.result_formula_text_list += ['(seg-{0}) ******'.format(k + 1)]

        # report start
        # tiltle
        field_title = '<< score field: [{}] >>\n' + '- -'*40 + '\n'
        _output_report_doc = field_title.format(field)

        # calculating for ratio and segment
        plist = self.input_score_ratio_cum
        _output_report_doc += '  raw score seg ratio: {}\n'.\
            format([format(plist[j]-plist[j-1] if j > 0 else plist[0], '0.4f')
                    for j in range(len(plist))])
        _output_report_doc += '  raw score cum ratio: {}\n'.\
            format([format(x, '0.4f') for x in self.input_score_ratio_cum])
        _output_report_doc += '  raw score set ratio: {}\n'.\
            format(self.result_ratio_dict[field])
        _output_report_doc += '  raw score endpoints: {}\n'.\
            format([x[1] for x in self.result_formula_coeff.values()])
        _output_report_doc += '  out score endpoints: {}\n'.\
            format([x[2] for x in self.result_formula_coeff.values()])

        # transforming formulas
        _output_report_doc += '- -'*40 + '\n'
        for i, fs in enumerate(self.result_formula_text_list):
            if i == 0:
                _output_report_doc += 'transforming formulas:\n'
            _output_report_doc += '                       {}\n'.format(fs)

        # statistics for raw and out score
        _output_report_doc += '- -'*40 + '\n'
        _output_report_doc += '           statistics: raw_mean={:2.2f}, raw_std={:2.2f}  ' \
                              'out_mean={:2.2f}, out_std={:2.2f}\n'.\
                              format(self.input_data[field].mean(), self.input_data[field].std(),
                                     self.output_data[field+'_plt'].mean(), self.output_data[field+'_plt'].std())

        # differece between raw and out score
        _diff_raw_out = self.output_data[field+'_plt']-self.output_data[field]
        _output_report_doc += 'score shift(out-raw): ' \
                              'shift_max={:3.1f}  ' \
                              'shift_min={:3.1f}  ' \
                              'shift_down_percent={:.2f}%\n'.\
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
        _output_report_doc += '   shift down segment: ' + str(_diff_list) + '\n'
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
        _output_report_doc += '       lower segments: ' + str(_diff_list) + '\n'
        _output_report_doc += '---'*40 + '\n'

        return _output_report_doc

    def report(self):
        print(self.output_report_doc)

    def plot(self, mode='model'):
        if mode not in ['raw', 'out', 'model', 'shift', 'dist', 'bar']:
            print('valid mode is: raw, out, model,shift, dist, bar')
            return
        if mode in 'shift, model':
            # mode: model describe the differrence of input and output score.
            self.__plot_model()
        elif mode in 'dist':
            self.__plot_dist()
        elif mode in 'bar':
            self.__plot_bar()
        elif not super(PltScore, self).plot(mode):
            print('\"{}\" is invalid'.format(mode))

    def __plot_bar(self):
        x = np.arange(self.input_score_max+1)
        for f in self.field_list:
            raw_label = [str(x) for x in self.map_table['seg']][::-1]
            raw_data = list(self.map_table[f+'_count'])[::-1]
            out_seg = run_seg(self.output_data,
                              [f+'_plt'],
                              segmax=self.output_score_max,
                              segmin=self.output_score_min)
            out_data = [0 for _ in raw_label]
            for ri, row in out_seg.output_data.iterrows():
                for i, s in enumerate(raw_label):
                    if int(s) == int(row['seg']):
                        out_data[i] = row[f+'_plt_count']
            fig, ax = pyplt.subplots()
            ax.set_xticks(x)
            ax.set_xticklabels(raw_label)
            ax.legend()
            width = 0.4
            rects1 = ax.bar(x - width / 2, raw_data, width, label=f)
            rects2 = ax.bar(x + width / 2, out_data, width, label=f+'_plt')
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rects in [rects1, rects2]:
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(int(height)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            fig.tight_layout()
            pyplt.show()

    def __plot_dist(self):
        for f in self.field_list:
            fig, ax = pyplt.subplots()
            x_data = list(self.map_table.seg)[::-1]
            ax.plot(x_data,
                    list(self.map_table[f + '_count'])[::-1],
                    'o-',
                    label='score:' + f)
            out_seg = run_seg(self.output_data,
                              [f+'_plt'],
                              segmax=self.output_score_max,
                              segmin=self.output_score_min)
            # out_data = [0 for _ in self.map_table['seg']]
            # for ri, row in out_seg.output_data.iterrows():
            #     for i, s in enumerate(x_data):
            #         if int(s) == int(row['seg']):
            #             out_data[i] = row[f+'_plt_count']
            #
            ax.plot(list(out_seg.output_data['seg'])[::-1],
                    list(out_seg.output_data[f+'_plt_count'])[::-1],
                    'o-',
                    label='score:' + f + '_plt')
            ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        # legend.get_frame().set_facecolor('C0')
        pyplt.show()

    def __plot_model(self):
        # 分段线性转换模型
        pyplt.rcParams['font.sans-serif'] = ['SimHei']
        pyplt.rcParams.update({'font.size': 8})
        for i, fs in enumerate(self.field_list):
            result = self.result_dict[fs]
            input_points = result['input_score_points']
            in_max = max(input_points)
            ou_min = min([min(p) for p in self.output_score_points])
            ou_max = max([max(p) for p in self.output_score_points])

            pyplt.figure(fs+'_plt')
            pyplt.rcParams.update({'font.size': 10})
            pyplt.title(u'转换模型({})'.format(fs))
            pyplt.xlim(min(input_points), max(input_points))
            pyplt.ylim(ou_min, ou_max)
            pyplt.xlabel(u'\n原始分数')
            pyplt.ylabel(u'转换分数')
            pyplt.xticks([])
            pyplt.yticks([])

            formula = self.result_dict[fs]['coeff']
            for cfi, cf in enumerate(formula.values()):
                x = cf[1] if self.mode_score_order in ['ascending', 'a'] else cf[1][::-1]
                y = cf[2] if self.mode_score_order in ['ascending', 'a'] else cf[2][::-1]
                pyplt.plot(x, y)
                for j in [0, 1]:
                    pyplt.plot([x[j], x[j]], [0, y[j]], '--')
                    pyplt.plot([0, x[j]], [y[j], y[j]], '--')
                for j, xx in enumerate(x):
                    pyplt.text(xx-1 if j == 1 else xx, ou_min-2, '{}'.format(int(xx)))
                for j, yy in enumerate(y):
                    pyplt.text(1, yy-2 if j == 1 else yy+1, '{}'.format(int(yy)))

            # darw y = x for showing score shift
            pyplt.plot((0, in_max), (0, in_max), 'ro-')

        pyplt.show()
        return

    def report_map_table(self):
        fs_list = ['seg']
        for ffs in self.field_list:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_plt']
        print(self.map_table[fs_list])


class Zscore(ScoreTransformModel):
    """
    transform raw score to Z-score according to percent position on normal cdf
    input data:
    rawdf = raw score dataframe
    stdNum = standard error numbers
    output data:
    output_data = result score with raw score field name + '_z'
    """
    # HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self):
        super(Zscore, self).__init__('zt')
        self.model_name = 'zscore'
        self.stdNum = 3
        self.maxRawscore = 150
        self.minRawscore = 0
        self.map_table = None
        self.output_data_decimal = 0
        # self.__currentfield = None

        # deprecated
        # create norm table
        self._samplesize = 100000    # cdf error is less than 0.0001
        self._normtable = None

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_para(self, std_num=3, rawscore_max=100, rawscore_min=0,
                       output_decimal=6):
        self.stdNum = std_num
        self.maxRawscore = rawscore_max
        self.minRawscore = rawscore_min
        self.output_data_decimal = output_decimal

    def check_parameter(self):
        if self.maxRawscore <= self.minRawscore:
            print('max raw score or min raw score error!')
            return False
        if self.stdNum <= 0:
            print('std number is error!')
            return False
        return True

    def run(self):
        # check data and parameter in super
        if not super(Zscore, self).run():
            return
        # create and calculate output_data
        print('start run...')
        # deprecated for time cost
        # if self._normtable is None:
        #     print('create norm table')
        #     self._normtable = get_norm_dist_table(self._samplesize, stdnum=4)
        #     self._normtable.loc[max(self._normtable.index), 'cdf'] = 1
        self.output_data = self.input_data.copy()
        self.map_table = \
            self.__get_map_table(self.output_data, self.maxRawscore, self.minRawscore, self.field_list)

        for sf in self.field_list:
            print('start run on field: {}...'.format(sf))
            st = time.clock()
            self._get_zscore_in_map_table(sf)
            df = self.output_data
            print('zscore calculating(1): create field {}_zscore ...'.format(sf))
            df.loc[:, sf+'_zscore'] = \
                df[sf].apply(lambda x: x if x in self.map_table.seg.values else -999)
            print('zscore calculating(1)...use time{}'.format(time.clock()-st))
            print('zscore calculating(2): get zscore from map_table...')
            df.loc[:, sf+'_zscore'] = \
                df[sf + '_zscore'].replace(self.map_table.seg.values,
                                           self.map_table[sf + '_zscore'].values)
            self.output_data = df
            print('{}_zscore finished with {} consumed'.format(sf, round(time.clock()-st, 2)))

    # new method for time reason
    def lookup_zscore(self, percent):
        low_z = -self.stdNum
        high_z = self.stdNum
        curr_z = low_z
        if sts.norm.cdf(low_z) >= percent:
            return round45r(low_z, self.output_data_decimal)
        elif sts.norm.cdf(high_z) <= percent:
            return round45r(high_z, self.output_data_decimal)
        err = 10**(-7)
        iter_num = 1000
        while True:
            iter_num = iter_num - 1
            curr_p = sts.norm.cdf(curr_z)
            if abs(curr_p - percent) < err:
                break
                # return curr_z
            if iter_num < 0:
                break
                # return curr_z
            if curr_p > percent:
                high_z = curr_z
            elif curr_p < percent:
                low_z = curr_z
            curr_z = (low_z + high_z) / 2
        return round45r(curr_z, self.output_data_decimal)

    def _get_zscore_in_map_table(self, sf):
        # use method lookup_zscore
        if sf+'_percent' in self.map_table.columns.values:
            self.map_table.loc[:, sf + '_zscore'] = \
                self.map_table[sf + '_percent'].apply(self.lookup_zscore)
        else:
            print('error: not found field{}+"_percent"!'.format(sf))
        # deprecated for time cost
        # if sf+'_percent' in self.map_table.columns.values:
        #     self.map_table.loc[:, sf + '_zscore'] = \
        #         self.map_table[sf + '_percent'].apply(self.__get_zscore_from_normtable)
        # else:
        #     print('error: not found field{}+"_percent"!'.format(sf))

    def __get_zscore_from_normtable(self, p):
        df = self._normtable.loc[self._normtable.cdf >= p - Zscore.MinError][['sv']].head(1).sv
        y = df.values[0] if len(df) > 0 else None
        if y is None:
            print('error: cdf value[{}] can not find zscore in normtable!'.format(p))
            return y
        return max(-self.stdNum, min(y, self.stdNum))

    @staticmethod
    def __get_map_table(df, maxscore, minscore, scorefieldnamelist):
        """no sort problem in this map_table usage"""
        seg = SegTable()
        seg.set_data(df, scorefieldnamelist)
        seg.set_para(segmax=maxscore, segmin=minscore, segsort='a')
        seg.run()
        return seg.output_data

    # deprecated for time cost
    @staticmethod
    def get_normtable(stdnum=4, precise=4):
        cdf_list = []
        sv_list = []
        pdf_list = []
        cdf0 = 0
        scope = stdnum * 2 * 10**precise + 1
        for x in range(scope):
            sv = -stdnum + x/10**precise
            cdf = sts.norm.cdf(sv)
            pdf = cdf - cdf0
            cdf0 = cdf
            pdf_list.append(pdf)
            sv_list.append(sv)
            cdf_list.append(cdf)
        return pd.DataFrame({'pdf': pdf_list, 'sv': sv_list, 'cdf': cdf_list})

    def report(self):
        if type(self.output_data) == pd.DataFrame:
            print('output score desc:\n', self.output_data.describe())
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('para:')
        print('\tzscore stadard diff numbers:{}'.format(self.stdNum))
        print('\tmax score in raw score:{}'.format(self.maxRawscore))
        print('\tmin score in raw score:{}'.format(self.minRawscore))

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super(Zscore, self).plot(mode)
        else:
            print('not support this mode!')


class Tscore(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模,平均数为50,标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出,是为了纪念推孟和桑代克
    对智力测验,尤其是提出智商这一概念所作出的巨大贡献。'''

    def __init__(self):
        super(Tscore, self).__init__('t')
        # self.model_name = 't'

        self.rscore_max = 150
        self.rscore_min = 0
        self.tscore_std = 10
        self.tscore_mean = 50
        self.tscore_stdnum = 4

        self.output_data_decimal = 0
        self.zscore_decimal = 6

        self.map_table = None

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_para(self, rawscore_max=150, rawscore_min=0,
                       tscore_mean=500, tscore_std=100, tscore_stdnum=4,
                       output_decimal=0):
        self.rscore_max = rawscore_max
        self.rscore_min = rawscore_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum
        self.output_data_decimal = output_decimal

    def run(self):
        zm = Zscore()
        zm.set_data(self.input_data, self.field_list)
        zm.set_para(std_num=self.tscore_stdnum,
                          rawscore_min=self.rscore_min,
                          rawscore_max=self.rscore_max,
                          output_decimal=self.zscore_decimal)
        zm.run()
        self.output_data = zm.output_data
        namelist = self.output_data.columns
        for sf in namelist:
            if '_zscore' in sf:
                newsf = sf.replace('_zscore', '_tscore')
                self.output_data.loc[:, newsf] = \
                    self.output_data[sf].apply(
                        lambda x: round45r(x * self.tscore_std + self.tscore_mean,
                                           self.output_data_decimal))
        self.map_table = zm.map_table

    def report(self):
        print('T-score by normal table transform report')
        print('-' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            print('    fields:', self.field_list)
            print(self.input_data[self.field_list].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.field_list]
            print('T-score desc:')
            print('    fields:', out_fields)
            print(self.output_data[out_fields].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('para:')
        print('\tzscore stadard deviation numbers:{}'.format(self.tscore_std))
        print('\tmax score in raw score:{}'.format(self.rscore_max))
        print('\tmin score in raw score:{}'.format(self.rscore_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super(Tscore, self).plot(mode)


class TscoreLinear(ScoreTransformModel):
    """Get Zscore by linear formula: (x-mean)/std"""
    def __init__(self):
        super(TscoreLinear, self).__init__('tzl')

        self.model_name = 'tzl'
        self.rawscore_max = 150
        self.rawscore_min = 0
        self.tscore_mean = 50
        self.tscore_std = 10
        self.tscore_stdnum = 4

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_para(self,
                       input_score_max=150,
                       input_score_min=0,
                       tscore_std=10,
                       tscore_mean=50,
                       tscore_stdnum=4):
        self.rawscore_max = input_score_max
        self.rawscore_min = input_score_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum

    def check_data(self):
        super(TscoreLinear, self).check_data()
        return True

    def check_parameter(self):
        if self.rawscore_max <= self.rawscore_min:
            print('raw score max and min error!')
            return False
        if self.tscore_std <= 0 | self.tscore_stdnum <= 0:
            print('t_score std number error:std={}, stdnum={}'.format(self.tscore_std, self.tscore_stdnum))
            return False
        return True

    def run(self):
        super(TscoreLinear, self).run()
        self.output_data = self.input_data
        for sf in self.field_list:
            rmean, rstd = self.output_data[[sf]].describe().loc[['mean', 'std']].values[:, 0]
            self.output_data[sf + '_zscore'] = \
                self.output_data[sf].apply(
                    lambda x: min(max((x - rmean) / rstd, -self.tscore_stdnum), self.tscore_stdnum))
            self.output_data.loc[:, sf + '_tscore'] = \
                self.output_data[sf + '_zscore'].\
                apply(lambda x: x * self.tscore_std + self.tscore_mean)

    def report(self):
        print('TZ-score by linear transform report')
        print('-' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            print(self.input_data[[f for f in self.field_list]].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            print('raw,T,Z score desc:')
            print(self.output_data[[f+'_tscore' for f in self.field_list]].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('para:')
        print('\tzscore stadard deviation numbers:{}'.format(self.tscore_std))
        print('\tmax score in raw score:{}'.format(self.rawscore_max))
        print('\tmin score in raw score:{}'.format(self.rawscore_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super(TscoreLinear, self).plot(mode)


class GradeScore(ScoreTransformModel):
    """
    grade score transform model
    default set to zhejiang project:
    grade_ratio_table = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
    grade_score_table = [100, 97, ..., 40]
    grade_order = 'd'   # d: from high to low, a: from low to high
    """
    def __init__(self):
        super(GradeScore, self).__init__('grade')
        __zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        self.mode_ratio_approx_set = 'upper_min, lower_max, near_max, near_min'

        self.input_score_max = 150
        self.input_score_min = 0
        self.ratio_grade_table = [round45r(sum(__zhejiang_ratio[0:j + 1]) * 0.01, 2)
                                  for j in range(len(__zhejiang_ratio))]
        self.grade_score_table = [100 - x * 3 for x in range(len(self.ratio_grade_table))]
        self.grade_no = [x for x in range(1, len(self.ratio_grade_table) + 1)]
        self.grade_order = 'd' if self.grade_score_table[0] > self.grade_score_table[-1] else 'a'
        self.mode_ratio_approx = 'near'

        self.map_table = None
        self.output_data = None
        self.report_doc = ''
        self.result_dict = dict()

    def set_data(self, input_data=None, field_list=None):
        if isinstance(input_data, pd.DataFrame):
            self.input_data = input_data
        if isinstance(field_list, list):
            self.field_list = field_list
        elif isinstance(field_list, str):
            self.field_list = [field_list]
        else:
            print('error field_list: {}'.format(field_list))

    def set_para(self,
                 maxscore=None,
                 minscore=None,
                 grade_ratio_table=None,
                 grade_score_table=None,
                 mode_ratio_approx=None,
                 mode_ratio_cumu=None
                 ):
        if isinstance(maxscore, int):
            if len(self.field_list) > 0:
                if maxscore >= max([max(self.input_data[f]) for f in self.field_list]):
                    self.input_score_max = maxscore
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set field_list first!')
        if isinstance(minscore, int):
            self.input_score_min = minscore
        if isinstance(grade_ratio_table, list) or isinstance(grade_ratio_table, tuple):
            self.ratio_grade_table = [round45r(1 - sum(grade_ratio_table[0:j + 1]) * 0.01, 2)
                                      for j in range(len(grade_ratio_table))]
            if sum(grade_ratio_table) != 100:
                print('ratio table is wrong, sum:{} is not 100!'.format(sum(grade_ratio_table)))
        if isinstance(grade_score_table, list) or isinstance(grade_score_table, tuple):
            self.grade_score_table = grade_score_table
        if len(self.ratio_grade_table) != len(self.grade_score_table):
            print('error grade data set, ratio/score table is not same length!')
            print(self.ratio_grade_table, '\n', self.grade_score_table)
        self.grade_no = [x+1 for x in range(len(self.ratio_grade_table))]
        self.grade_order = 'd' if self.grade_score_table[0] > self.grade_score_table[-1] else 'a'
        self.ratio_grade_table = [1] + self.ratio_grade_table
        if mode_ratio_approx in self.mode_ratio_approx_set:
            self.mode_ratio_approx = mode_ratio_approx

    def run(self):
        if len(self.field_list) == 0:
            print('to set field_list first!')
            return
        print("--- grade Score Transform Start ---")
        t0 = time.time()

        # create seg-percent map table
        seg = SegTable()
        seg.set_data(input_data=self.input_data,
                     field_list=self.field_list)
        seg.set_para(segmax=self.input_score_max,
                           segmin=self.input_score_min,
                           segsort=self.grade_order)
        seg.run()
        self.map_table = seg.output_data

        # key step: create grade score map list
        # self.__get_grade_table()  # depricated
        self.get_grade_map_by_approx()  # new method by approx

        # make output_data by map
        self.output_data = self.input_data
        self.report_doc = {}
        for sf in self.field_list:
            print('transform score at field:{} ...'.format(sf))
            dft = self.output_data

            # get percent
            dft[sf+'_percent'] = dft.loc[:, sf].replace(
                self.map_table['seg'].values, self.map_table[sf+'_percent'].values)

            # get grade no by map
            dft.loc[:, sf+'_grade'] = dft.loc[:, sf].replace(
                self.map_table['seg'].values, self.map_table[sf + '_grade'].values)

            # get grade score by map
            dft.loc[:, sf+'_grade_score'] = \
                dft.loc[:, sf+'_grade'].apply(lambda x: self.grade_score_table[int(x)-1]if x > 0 else x)
            if self.output_data_decimal == 0:
                # format to int
                dft = dft.astype({sf+'_grade': int, sf+'_grade_score': int})

            # save to output_data
            self.output_data = dft

            # save to report_doc
            grade_max = self.map_table.groupby(sf+'_grade')['seg'].max()
            grade_min = self.map_table.groupby(sf+'_grade')['seg'].min()
            self.report_doc.update({sf: ['grade({}):{}-{}'.format(j+1, x[0], x[1])
                                         for j, x in enumerate(zip(grade_max, grade_min))]})
            pt = pd.pivot_table(self.map_table, values='seg', index=sf + '_grade', aggfunc=[max, min])
            self.result_dict.update({sf: [(idx, (pt.loc[idx, ('min', 'seg')], pt.loc[idx, ('max', 'seg')]))
                                          for idx in pt.index]})
        # running end
        print('used time:{:6.4f}'.format(time.time()-t0))
        print('---grade Score Transform End---')

    # make map seg_percent to grade in map_table, used to calculate grade score
    def __get_grade_table(self):
        for sf in self.field_list:
            self.map_table.loc[:, sf+'_grade'] = self.map_table[sf+'_percent'].\
                apply(lambda x: self.__map_percent_to_grade(1 - x))
            self.map_table.astype({sf+'_grade': int})
            pt = pd.pivot_table(self.map_table, values='seg', index=sf+'_grade', aggfunc=[max, min])
            self.result_dict.update({sf: [(idx, (pt.loc[idx, ('min', 'seg')], pt.loc[idx, ('max', 'seg')]))
                                          for idx in pt.index]}
                                    )

    def __map_percent_to_grade(self, p):
        # p_start = 0 if self.grade_order in 'a, ascending' else 1
        for j in range(len(self.ratio_grade_table)-1):
            # logic = (p_start <= p <= r) if self.grade_order in 'a, ascending' else (r <= p <= p_start)
            if self.ratio_grade_table[j] >= p >= self.ratio_grade_table[j+1]:
                return self.grade_no[j]
        print('percent:{} not found in {}'.format(p, self.ratio_grade_table))
        return self.grade_no[-1]

    # calculate grade score by mode_ratio_approx
    def get_grade_map_by_approx(self):
        ratio_table = [1-x for x in self.ratio_grade_table[1:]]
        for sf in self.field_list:
            self.map_table.loc[:, sf+'_grade'] = self.map_table[sf+'_percent'].apply(lambda x: 1)
            self.map_table.astype({sf+'_grade': int})
            last_p = 0
            curr_grade_no = self.grade_no[0]
            curr_grade_ratio = ratio_table[0]
            curr_grade_score = self.grade_score_table[0]
            max_count = self.map_table['seg'].count()
            for ri, rv in self.map_table.iterrows():
                self.map_table.loc[ri, sf + '_grade'] = curr_grade_no
                self.map_table.loc[ri, sf + '_grade_score'] = curr_grade_score
                if rv[sf+'_percent'] == 1:  # set to end grade and score
                    self.map_table.loc[ri, sf+'_grade'] = self.grade_no[-1]
                    self.map_table.loc[ri, sf+'_grade_score'] = self.grade_score_table[-1]
                    continue
                curr_p = rv[sf+'_percent']
                if curr_p >= curr_grade_ratio:
                    curr_to_new_grade = False
                    d1 = abs(curr_grade_ratio - last_p)
                    d2 = abs(curr_grade_ratio - curr_p)
                    if d1 < d2:
                        if self.mode_ratio_approx in 'upper_min, near':
                            curr_to_new_grade = True
                    elif d1 == d2:
                        if self.mode_ratio_approx in 'upper_min, near_min, near_min':
                            curr_to_new_grade = True
                    else:  # d2 < d1
                        if self.mode_ratio_approx in 'upper_min':
                            curr_to_new_grade = True
                    if curr_to_new_grade:
                        curr_grade_no += 1
                        curr_grade_ratio = ratio_table[curr_grade_no-1]
                        curr_grade_score = self.grade_score_table[curr_grade_no - 1]
                        self.map_table.loc[ri, sf + '_grade'] = curr_grade_no
                        self.map_table.loc[ri, sf + '_grade_score'] = curr_grade_score
                    else:
                        self.map_table.loc[ri, sf+'_grade'] = curr_grade_no
                        self.map_table.loc[ri, sf+'_grade_score'] = curr_grade_score
                        if ri < max_count & self.map_table.loc[ri+1, sf+'_count'] > 0:
                            curr_grade_no += 1
                            curr_grade_ratio = ratio_table[curr_grade_no-1]
                            curr_grade_score = self.grade_score_table[curr_grade_no - 1]
                last_p = curr_p
            if self.output_data_decimal == 0:
                self.map_table = self.map_table.astype({sf+'_grade_score': int})

    def report(self):
        print('grade-score Transform Report')
        p_ = False
        for sf in self.field_list:
            if p_:
                print('-' * 50)
            else:
                print('=' * 50)
                p_ = True
            print('<<{}>> grade No: Raw_Score_Range'.format(sf))
            for k in self.result_dict[sf]:
                print('    grade {no:>2}: [{int_min:>3},{int_max:>3}]'.
                      format(no=str(k[0]), int_min=k[1][0], int_max=k[1][1]))
        print('=' * 50)

    def plot(self, mode='raw'):
        super(GradeScore, self).plot(mode)

    def check_parameter(self):
        if self.input_score_max > self.input_score_min:
            return True
        else:
            print('raw score max value is less than min value!')
        return False

    def check_data(self):
        return super(GradeScore, self).check_data()

    def report_map_table(self):
        fs_list = ['seg']
        for ffs in self.field_list:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_grade']
        df = self.map_table.copy()
        for fs in fs_list:
            if 'percent' in fs:
                df[fs] = df[fs].apply(lambda x: round(x, 8))
        # print(ptt.make_page(df=df[fs_list],
        #                     title='grade score table for {}'.format(self.model_name),
        #                     pagelines=self.input_score_max+1))
        print(df[fs_list])


class GradeScoreTao(ScoreTransformModel):
    """
    grade Score model from Tao BaiQiang
    top_group = df.sort_values(field,ascending=False).head(int(df.count(0)[field]*0.01))[[field]]
    high_grade = top_group[field].describe().loc['mean', field]
    intervals = [minscore, high_grade*1/50], ..., [high_grade, max_score]
    以原始分值切分，形成的分值相当于等距合并，粒度直接增加
    实质上失去了等级分数的意义
    本模型仍然存在高分区过度合并问题
    """

    def __init__(self):
        super(GradeScoreTao, self).__init__('grade')
        self.model_name = 'taobaiqiang'

        self.grade_num = 50
        self.input_score_max = 100
        self.input_score_min = 0
        self.max_ratio = 0.01  # 1%
        self.input_data = pd.DataFrame()

        self.grade_no = [x for x in range(self.grade_num+1)]
        self.map_table = None
        self.grade_dist_dict = {}  # fs: grade_list, from max to min
        self.output_data = pd.DataFrame()

    def set_data(self, input_data=pd.DataFrame(), field_list=None):
        if len(input_data) > 0:
            self.input_data = input_data
        if isinstance(field_list, list) or isinstance(field_list, tuple):
            self.field_list = field_list

    def set_para(self,
                       maxscore=None,
                       minscore=None,
                       grade_num=None,
                       ):
        if isinstance(maxscore, int):
            if len(self.field_list) > 0:
                if maxscore >= max([max(self.input_data[f]) for f in self.field_list]):
                    self.input_score_max = maxscore
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set field_list first!')
        if isinstance(minscore, int):
            self.input_score_min = minscore
        if isinstance(grade_num, int):
            self.grade_num = grade_num
        self.grade_no = [x for x in range(self.grade_num+1)]

    def run(self):
        self.run_create_grade_dist_list()
        self.run_create_output_data()

    def run_create_grade_dist_list(self):
        # mode_ratio_approx = 'near'
        seg = SegTable()
        seg.set_para(segmax=self.input_score_max,
                     segmin=self.input_score_min,
                     segsort='d')
        seg.set_data(self.input_data,
                     self.field_list)
        seg.run()
        self.map_table = seg.output_data
        for fs in self.field_list:
            lastpercent = 0
            lastseg = self.input_score_max
            for ind, row in self.map_table.iterrows():
                curpercent = row[fs + '_percent']
                curseg = row['seg']
                if row[fs+'_percent'] > self.max_ratio:
                    if curpercent - self.max_ratio > self.max_ratio - lastpercent:
                        max_score = lastseg
                    else:
                        max_score = curseg
                    max_point = self.input_data[self.input_data[fs] >= max_score][fs].mean()
                    # print(fs, max_score, curseg, lastseg)
                    self.grade_dist_dict.update({fs: round45r(max_point/self.grade_num, 8)})
                    break
                lastpercent = curpercent
                lastseg = curseg

    def run_create_output_data(self):
        dt = copy.deepcopy(self.input_data[self.field_list])
        for fs in self.field_list:
            dt.loc[:, fs+'_grade'] = dt[fs].apply(lambda x: self.run__get_grade_score(fs, x))
            dt2 = self.map_table
            dt2.loc[:, fs+'_grade'] = dt2['seg'].apply(lambda x: self.run__get_grade_score(fs, x))
            self.output_data = dt

    def run__get_grade_score(self, fs, x):
        if x == 0:
            return x
        grade_dist = self.grade_dist_dict[fs]
        for i in range(self.grade_num):
            minx = i * grade_dist
            maxx = (i+1) * grade_dist if i < self.grade_num-1 else self.input_score_max
            if minx < x <= maxx:
                return i+1
        return -1

    def plot(self, mode='raw'):
        pass

    def report(self):
        print(self.output_data[[f+'_grade' for f in self.field_list]].describe())

    def print_map_table(self):
        # print(ptt.make_mpage(self.map_table))
        print(self.map_table)


# version 1.0.1 2018-09-24
class SegTable(object):
    """
    * 计算pandas.DataFrame中分数字段的分段人数表
    * segment table for score dataframe
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    * from 09-17-2017

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(input_data:DataFrame, field_list:list)
        input_data: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        field_list: list, field names used to calculate seg table, empty for calculate all fields
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
    output_data: 输出分段数据
            seg: seg value
        [field]: field name in field_list
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
        print(seg.output_data.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          field_list type is digit, for example: int or float

        3)可以单独设置数据(input_data),字段列表（field_list),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.field_list = ['score_1', 'score_2'];
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
        self.__input_dataframe = None
        self.__segFields = []
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
        self.__output_dataframe = None
        # run status
        self.__run_completed = False

    @property
    def output_data(self):
        return self.__output_dataframe

    @property
    def input_data(self):
        return self.__input_dataframe

    @input_data.setter
    def input_data(self, df):
        self.__input_dataframe = df

    @property
    def field_list(self):
        return self.__segFields

    @field_list.setter
    def field_list(self, field_list):
        self.__segFields = field_list

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

    def set_data(self, input_data, field_list=None):
        self.input_data = input_data
        if type(field_list) == str:
            field_list = [field_list]
        if (not isinstance(field_list, list)) & isinstance(input_data, pd.DataFrame):
            self.field_list = input_data.columns.values
        else:
            self.field_list = field_list
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
        print('    use seglist:{0}'.format(self.__useseglist, self.__segList))
        print('        seglist:{1}'.format(self.__useseglist, self.__segList))
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
        if isinstance(self.__input_dataframe, pd.Series):
            self.__input_dataframe = pd.DataFrame(self.__input_dataframe)
        if not isinstance(self.__input_dataframe, pd.DataFrame):
            print('error: raw score data is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('error: segmax({}) is not greater than segmin({})!'.format(self.__segMax, self.__segMin))
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('error: segstep({}) is too small or big!'.format(self.__segStep))
            return False
        if not isinstance(self.field_list, list):
            if isinstance(self.field_list, str):
                self.field_list = [self.field_list]
            else:
                print('error: segfields type({}) error.'.format(type(self.field_list)))
                return False

        for f in self.field_list:
            if f not in self.input_data.columns:
                print("error: field('{}') is not in input_data fields({})".
                      format(f, self.input_data.columns.values))
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
        self.__output_dataframe = pd.DataFrame({'seg': seglist})
        outdf = self.__output_dataframe
        for f in self.field_list:
            # calculate preliminary group count
            tempdf = self.input_data
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

            # self.__output_dataframe = outdf.copy()
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
        self.__output_dataframe = outdf
        return

    def __run_special_step(self, field: str):
        """
        processing count for step > 1
        :param field: for seg stepx
        :return: field_countx in output_data
        """
        f = field
        segcountname = f + '_count{0}'.format(self.__segStep)
        self.__output_dataframe[segcountname] = np.int64(-1)
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
        for index, row in self.__output_dataframe.iterrows():
            cum += row[f + '_count']
            curseg = np.int64(row['seg'])
            if curseg in [self.__segMax, self.__segMin]:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if (self.__segStart <= self.__segMin) | (self.__segStart >= self.__segMax):
                    curpoint += curstep
                continue
            if curseg in [self.__segStart, curpoint]:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
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
        self.__output_dataframe[segcountname] = np.int64(-1)
        segpoint = sorted(self.__segList) \
            if self.__segSort.lower() == 'a' \
            else sorted(self.__segList)[::-1]
        # curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        # curpoint = self.__segStart
        cum = 0
        pos = 0
        curpoint = segpoint[pos]
        rownum = len(self.__output_dataframe)
        cur_row = 0
        lastindex = 0
        maxpoint = max(self.__segList)
        minpoint = min(self.__segList)
        list_sum = 0
        self.__output_dataframe.loc[:, f+'_list_sum'] = 0
        for index, row in self.__output_dataframe.iterrows():
            curseg = np.int64(row['seg'])
            # cumsum
            if self.__usealldata | (minpoint <= curseg <= maxpoint):
                cum += row[f + '_count']
                list_sum += row[f+'_count']
                self.__output_dataframe.loc[index, f+'_list_sum'] = np.int64(list_sum)
            # set to seg count, only set seg in seglist
            if curseg == curpoint:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if pos < len(segpoint)-1:
                    pos += 1
                    curpoint = segpoint[pos]
                else:
                    lastindex = index
            elif cur_row == rownum:
                if self.__usealldata:
                    self.__output_dataframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1

    def plot(self):
        if not self.__run_completed:
            if self.__display:
                print('result is not created, please run!')
            return
        legendlist = []
        step = 0
        for sf in self.field_list:
            step += 1
            legendlist.append(sf)
            pyplt.figure('map_table figure({})'.
                       format('Descending' if self.__segSort in 'aA' else 'Ascending'))
            pyplt.subplot(221)
            pyplt.hist(self.input_data[sf], 20)
            pyplt.title('histogram')
            if step == len(self.field_list):
                pyplt.legend(legendlist)
            pyplt.subplot(222)
            pyplt.plot(self.output_data.seg, self.output_data[sf+'_count'])
            if step == len(self.field_list):
                pyplt.legend(legendlist)
            pyplt.title('distribution')
            pyplt.xlim([self.__segMin, self.__segMax])
            pyplt.subplot(223)
            pyplt.plot(self.output_data.seg, self.output_data[sf + '_sum'])
            pyplt.title('cumsum')
            pyplt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                pyplt.legend(legendlist)
            pyplt.subplot(224)
            pyplt.plot(self.output_data.seg, self.output_data[sf + '_percent'])
            pyplt.title('percentage')
            pyplt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                pyplt.legend(legendlist)
            pyplt.show()
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
