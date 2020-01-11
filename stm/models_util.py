# coding: utf-8


"""
    plot_models(font_size=12)
    山东、浙江、上海、北京、天津、广东、湖南方案等级转换分数分布直方图
    plot models distribution hist graph including shandong, zhejiang, shanghai, beijing, tianjin

    round45r(v: float, dec = 0)
    四舍五入函数, 用于改进round产生的偶数逼近和二进制表示方式产生的四舍五入误差
    function for rounding strictly at some decimal position
          v： 输入浮点数
        dec： 保留小数位数

    get_norm_table(size=400, std=1, mean=0, stdnum=4)
    生成具有指定记录数（size = 400）、标准差(std=1)、均值(mean=0)、截止标准差数（最小最大）(stdnum=4)的正态分布表
    create norm data dataframe with assigned scale, mean, standard deviation, std range
    return DataFrame({'rv': random var, 'cdf': cdf , 'pdf': pdf}
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
import numbers
from stm import models_sys as mdin


def show_models():
    for k in mdin.Models:
        v = mdin.Models[k]
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
            x_data = [np.mean(x) for x in mdin.Models[k].section][::-1]
            _wid = 10
        elif k in ['ss7']:
            x_data = [np.mean(x) for x in mdin.Models[k].section][::-1]
            _wid = 10
        elif k in ['hn900']:
            x_data = [x for x in range(100, 901)]
            _wid = 1
        elif k in ['hn300']:
            x_data = [x for x in range(60, 301)]
            _wid = 1
        else:
            raise ValueError(k)
        plot.bar(x_data, mdin.Models[k].ratio[::-1], width=_wid)
        plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))


def plot_norm_test(df, cols):
    for col in cols:
        _len = len(df)
        x = sorted(df[col])
        x1 = [np.log(v) for v in x]
        y = [(_i-0.375)/(_len+0.25) for _i in range(1, _len+1)]
        fig, ax = plot.subplots()
        ax.set_title('norm test')
        ax.plot(y, x1, 'o-', label='score:' + col)


def model_describe(name='shandong'):
    __ratio = mdin.Models[name].ratio
    __section = mdin.Models[name].section
    if name == 'hn900':
        __mean, __std, __skewness = 500, 100, 0
    elif name == 'hn300':
        __mean, __std, __skewness = 180, 30, 0
    else:
        samples = []
        [samples.extend([np.mean(s)]*int(__ratio[i])) for i, s in enumerate(__section)]
        __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))
    return __mean, __std, __skewness


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def get_norm_table(size=400, std=1, mean=0, stdnum=4):
    """
    function
        生成正态分布量表
        create normal distributed data(pdf,cdf) with preset std,mean,samples size
        变量区间： [-stdNum * std, std * stdNum]
        interval: [-stdNum * std, std * stdNum]
    parameters
        变量取值数 size: variable value number for create normal distributed PDF and CDF
        分布标准差  std: standard difference
        分布均值   mean: mean value
        标准差数 stdnum: used to define data range [-std * stdNum, std * stdNum]
    return
        DataFrame:   sv:stochastic variable value,
                    pdf: pdf value, 'cdf': cdf value
    """
    interval = [mean - std * stdnum, mean + std * stdnum]
    step = (2 * std * stdnum) / size
    varset = [mean + interval[0] + v*step for v in range(size+1)]
    cdflist = [sts.norm.cdf(v) for v in varset]
    pdflist = [sts.norm.pdf(v) for v in varset]
    ndf = pd.DataFrame({'rv': varset, 'cdf': cdflist, 'pdf': pdflist})
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
