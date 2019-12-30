# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
import numbers
from stm import modelsetin as msetin, modelext as mext


def show_models():
    for k in msetin.Models:
        v = msetin.Models[k]
        print('{:<20s} {},  {} '.format(k, v.type, v.desc))
        print('{:<20s} {}'.format(' ', v.ratio))
        print('{:<20s} {}'.format('', v.section))


def plot_models(font_size=12):
    _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'ss7', 'hn900']
    ms_dict = dict()
    for _name in _names:
        ms_dict.update({_name: model_describe(name=_name)})

    plot.figure('New Gaokao Score mcf.Models: name(mean, std, skewness)')
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
            x_data = [np.mean(x) for x in msetin.Models[k].section][::-1]
            _wid = 10
        elif k in ['ss7']:
            x_data = [np.mean(x) for x in msetin.Models[k].section][::-1]
            _wid = 10
        elif k in ['hn900']:
            x_data = [x for x in range(100, 901)]
            _wid = 1
        elif k in ['hn300']:
            x_data = [x for x in range(60, 301)]
            _wid = 1
        else:
            raise ValueError(k)
        plot.bar(x_data, msetin.Models[k].ratio[::-1], width=_wid)
        plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))


def model_describe(name='shandong'):
    __ratio = msetin.Models[name].ratio
    __section = msetin.Models[name].section
    if name == 'hn900':
        __mean, __std, __skewness = 500, 100, 0
    elif name == 'hn300':
        __mean, __std, __skewness = 180, 30, 0
    else:
        samples = []
        [samples.extend([np.mean(s)]*int(__ratio[i])) for i, s in enumerate(__section)]
        __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))
    return __mean, __std, __skewness


def check_model(model_name, model_lib=msetin.Models):
    if model_name in model_lib.keys():
        if not check_model_para(
            model_lib[model_name].type,
            model_lib[model_name].ratio,
            model_lib[model_name].section,
            model_lib[model_name].desc
        ):
            return False
    else:
        return False
    return True


def check_model_para(
                model_type='plt',
                model_ratio=None,
                model_section=None,
                model_desc=''
                ):
    if model_type not in ['ppt', 'plt', 'pgt']:
        print('error type: valid type must be in {}'.format(model_type, ['ppt', 'plt', 'pgt']))
        return False
    if len(model_ratio) != len(model_section):
        print('error length: the length of ratio group is not same as section group length !')
        return False
    for s in model_section:
        if len(s) > 2:
            print('error section: section must have 2 endpoints, real value: {}'.format(s))
            return False
        if s[0] < s[1]:
            print('error order: section endpoint order must be from large to small, '
                  'there: p1({}) < p2({})'.format(s[0], s[1]))
            return False
    if abs(sum(model_ratio) - 100) > 10**-12:
        print('error ratio: the sum of ratio must be 100, real sum={}!'.format(sum(model_ratio)))
        return False
    if not isinstance(model_desc, str):
        print('error desc: model desc(ription) must be str, but real type={}'.format(type(model_desc)))
    if model_type == 'ppt':
        if not all([x == y for x, y in model_section]):
            print('error section: ppt section, two endpoints must be same value!')
            return False
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
    ):

    st ={'mode_ratio_prox': mode_ratio_prox,
         'mode_ratio_cumu':mode_ratio_cumu,
         'mode_sort_order': mode_sort_order,
         'mode_section_point_first': mode_section_point_first,
         'mode_section_point_start': mode_section_point_start,
         'mode_section_point_last': mode_section_point_last,
         'mode_section_degraded': mode_section_degraded,
         'mode_section_lost': mode_section_lost,
         }
    for sk in st.keys():
        if sk in msetin.Strategy.keys():
            if not st[sk] in msetin.Strategy[sk]:
                print('error mode: {}={} not in {}'.format(sk, st[sk], msetin.Strategy[sk]))
                return False
        else:
            print('error mode: {} is not in Strategy-dict!'.format(sk))
            return False
    return True


def check_data(df=None, cols=None, raw_score_range=None):
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, pd.Series):
            print('warning: df is pandas.Series!')
            return False
        else:
            print('error data: df is not pandas.DataFrame!')
            return False
    if len(df) == 0:
        print('error data: df is empty!')
        return False
    if type(cols) not in (list, tuple):
        print('error type: cols must be list or tuple, real type is {}!'.format(type(cols)))
        return False
    for col in cols:
        if type(col) is not str:
            print('error col: {} is not str!'.format(col))
            return False
        else:
            if col not in df.columns:
                print('error col: {} is not in df.columns!'.format(col))
                return False
            if not isinstance(df[col][0], numbers.Number):
                print('type error: column[{}] not Number type!'.format(col))
                return False
            if df[col].min() < min(raw_score_range):
                print('warning: some scores in col={} not in raw score range:{}'.format(col, raw_score_range))

    return True


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def get_norm_table(size=400, std=1, mean=0, stdnum=4):
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


# test dataset
class TestData:
    """
    生成具有正态分布的数据，类型为 pandas.DataFrame, 列名为 sv
    create a score dataframe with fields 'score', used to test some application
    :__init__:parameter
        mean: 均值， std:标准差， max:最大值， min:最小值， size:行数
    :df
        DataFrame, columns = {'ksh', 'km1', 'km2'}
    """
    def __init__(self, mean=60, std=18, size=100000, max=100, min=0, decimals=0, dist='norm'):
        self.df = None
        self.df_mean = mean
        self.df_max = max
        self.df_min = min
        self.df_std = std
        self.df_size = size
        self.decimals=decimals
        self.dist = dist
        self.__make_data()

    def __make_data(self):
        self.df = pd.DataFrame({
            'ksh': ['37'+str(x).zfill(7) for x in range(1, self.df_size+1)],
            'km1': self.get_score(),
            'km2': self.get_score(),
        })

    def get_score(self):
        print('create score...')

        if self.decimals == 0:
            myround = lambda x: int(x)
        else:
            myround = lambda x: round(x, ndigits=self.decimals)
        norm_list = None
        if self.dist == 'norm':
            norm_list = sts.norm.rvs(loc=self.df_mean, scale=self.df_std, size=self.df_size)
            norm_list = np.array([myround(x) for x in norm_list])
            norm_list[np.where(norm_list > self.df_max)] = self.df_max
            norm_list[np.where(norm_list < self.df_min)] = self.df_min
            norm_list = norm_list.astype(np.int)
        return norm_list

    def __call__(self):
        return self.df
