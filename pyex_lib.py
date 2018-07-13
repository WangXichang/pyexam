# -*- coding: utf-8 -*-
# version 2017-09-16
# 2017-11-18


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def create_norm_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000):
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
    df = pd.DataFrame({'sv': [max(minvalue, min(x, maxvalue)) for x in np.random.normal(mean, std, size)]})
    return df


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def create_normaltable(size=400, std=1, mean=0, stdnum=4):
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
    cdflist = [stats.norm.cdf(v) for v in varset]
    pdflist = [stats.norm.pdf(v) for v in varset]
    ndf = pd.DataFrame({'sv': varset, 'cdf': cdflist, 'pdf':pdflist})
    return ndf


# use scipy.stats descibe report dataframe info
def report_stats_describe(dataframe, decdigits=4):
    """
    report statistic describe of a dataframe, with decimal digits = decnum
    峰度（Kurtosis）与偏态（Skewness）是量测数据正态分布特性的两个指标。
    峰度衡量数据分布的平坦度（flatness）。尾部大的数据分布峰度值较大。正态分布的峰度值为3。
        Kurtosis = 1/N * Sigma(Xi-Xbar)**4 / (1/N * Sigma(Xi-Xbar)**2)**2
    偏态量度对称性。0 是标准对称性正态分布。右（正）偏态表明平均值大于中位数，反之为左（负）偏态。
        Skewness = 1/N * Sigma(Xi-Xbar)**3 / (1/N * Sigma(Xi-Xbar)**2)**3/2
    :param
        dataframe: pandas DataFrame, raw data
        decnum: decimal number in report print
    :return(print)
        records
        min,max
        mean
        variance
        skewness
        kurtosis
    """

    def toround(listvalue, getdecdigits):
        return ''.join([('{:' + '1.' + str(getdecdigits)+'f}  ').format(round(x, getdecdigits)) for x in listvalue])

    def tosqrt(listvalue, getdecdigits):
        return ''.join([('{:1.'+str(getdecdigits)+'f}  ').format(round(np.sqrt(x), getdecdigits)) for x in listvalue])

    pr = [[stats.pearsonr(dataframe[x], dataframe[y])[0]
           for x in dataframe.columns] for y in dataframe.columns]
    sd = stats.describe(dataframe)
    cv = dataframe.cov()
    print('\trecords: ', sd.nobs)
    print('\tpearson recorrelation:')
    for i in range(len(dataframe.columns)):
        print('\t', toround(pr[i], 4))
    print('\tcovariance matrix:')
    for j in range(len(cv)):
        print('\t', toround(cv.iloc[j, :], 4))
    print('\tmin : ', toround(sd.minmax[0], 4))
    print('\tmax : ', toround(sd.minmax[1], 4))
    print('\tmean: ', toround(sd.mean, decdigits))
    print('\tvar : ', toround(sd.variance, decdigits))
    print('\tstd : ', tosqrt(sd.variance, decdigits))
    print('\tskewness: ', toround(sd.skewness, decdigits))
    print('\tkurtosis: ', toround(sd.kurtosis, decdigits))
    dict = {'record': sd.nobs,
            'max': sd.minmax[1],
            'min': sd.minmax[0],
            'mean': sd.mean,
            'var': sd.variance,
            'cov': cv,
            'cor': pr,
            'skewness': sd.skewness,
            'kurtosis': sd.kurtosis,
            }
    return dict

class ScoreStats:
    """
    :input
        input_data: score dataframe
    :result data
        output_data: segment for some fields
    :stats_fun
        report_stats: stats result for sdf data, max,mean,min,skew,kurtosis
        plot_line: plot distribute line graph
        plot_scatter: plot scatter graph

    """

    def __init__(self, input_data, field_list):
        self.df = input_data
        self.field_list = field_list
        self.peak_points_dict = {}      # field_name: (score_value, count_value, percent_value)
        self.std_at_percent = {}        # field_name: (percent, std_predict)

    def get_peak_point(self):
        pass


def pearson_relation(x, y):
    plt.scatter(x, y)
    return stats.pearsonr(x, y)[0]


def float_str(x, intlen, declen):
    x = round45(x,declen)
    fs = '{:'+str(intlen+declen+1)+'.'+str(declen)+'f}'
    return fs.format(x)


def int_str(x, len):
    return ('{:'+str(len)+'d}').format(x)


def df_format_digit2str(dfsource, intlen=2, declen=4, strlen=8):
    df = dfsource[[dfsource.columns[0]]]
    fdinfo = dfsource.dtypes
    for fs in fdinfo.index:
        if fdinfo[fs] in [np.float, np.float16, np.float32, np.float64]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: float_str(x, intlen, declen))
        elif fdinfo[fs] in [np.int, np.int8, np.int16, np.int32, np.int64]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: int_str(x, 6))
        elif fdinfo[fs] in [str]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: x.rjust(strlen))
    df.sort_index(axis=1)
    return df


def round45(v: float, i=0):
    vl = str(v).split('.')
    sign = -1 if v < 0 else 1
    if len(vl) == 2:
        if len(vl[1]) > i:
            if vl[1][i] >= '5':
                if i > 0:
                    vp = (eval(vl[1][:i]) + 1)/10**i * sign
                    return int(v)+vp
                else:
                    return int(v)+sign
            else:
                if i > 0:
                    return int(v) + eval(vl[1][:i])/10**i * sign
                else:
                    return int(v)
        else:
            return v
    return int(v)


def round45i(v:float, dec=0):
    u = int(v * 10**dec*10)
    return (int(u/10) + (1 if v >0 else -1))/10**dec if (abs(u) % 10 >= 5) else int(u/10)/10**dec
