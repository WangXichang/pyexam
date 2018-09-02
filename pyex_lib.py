# -*- coding: utf-8 -*-
# version 2017-09-16
# 2017-11-18


import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import random
from scipy import stats
from statsmodels.stats.diagnostic import lillifors


def exp_norm_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000, decimal=6):
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
                                  min(fun_round45i(x, decimal) if decimal > 0 else int(fun_round45i(x, decimal)),
                                      maxvalue))
                              for x in np.random.normal(mean, std, size)]})
    return df


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def exp_norm_table(size=400, std=1, mean=0, stdnum=4):
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
    ndf = pd.DataFrame({'sv': varset, 'cdf': cdflist, 'pdf': pdflist})
    return ndf


# use scipy.stats descibe report dataframe info
def report_describe(dataframe, decdigits=4):
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

    def fun_list2str(listvalue, decimal):
        return ''.join([('{:' + '1.' + str(decimal) + 'f}  ').
                       format(fun_round45i(x, decimal)) for x in listvalue])

    def fun_list2sqrt2str(listvalue, decimal):
        return fun_list2str([np.sqrt(x) for x in listvalue], decimal)

    pr = [[stats.pearsonr(dataframe[x], dataframe[y])[0]
           for x in dataframe.columns] for y in dataframe.columns]
    sd = stats.describe(dataframe)
    cv = dataframe.cov()
    print('\trecords: ', sd.nobs)
    print('\tpearson recorrelation:')
    for i in range(len(dataframe.columns)):
        print('\t', fun_list2str(pr[i], 4))
    print('\tcovariance matrix:')
    for j in range(len(cv)):
        print('\t', fun_list2str(cv.iloc[j, :], 4))
    print('\tmin : ', fun_list2str(sd.minmax[0], 4))
    print('\tmax : ', fun_list2str(sd.minmax[1], 4))
    print('\tmean: ', fun_list2str(sd.mean, decdigits))
    print('\tvar : ', fun_list2str(sd.variance, decdigits))
    print('\tstd : ', fun_list2sqrt2str(sd.variance, decdigits))
    print('\tskewness: ', fun_list2str(sd.skewness, decdigits))
    print('\tkurtosis: ', fun_list2str(sd.kurtosis, decdigits))
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


def plot_pearson_relation(x, y):
    plt.scatter(x, y)
    return stats.pearsonr(x, y)[0]


def fun_float2str(x, intlen, declen):
    x = fun_round45i(x, declen)
    fs = '{:'+str(intlen+declen+1)+'.'+str(declen)+'f}'
    return fs.format(x)


def fun_int2str(x, dec):
    return ('{:'+str(dec)+'d}').format(x)


def df_format_digit2str(dfsource, intlen=2, declen=4, strlen=8):
    df = dfsource[[dfsource.columns[0]]]
    fdinfo = dfsource.dtypes
    for fs in fdinfo.index:
        if fdinfo[fs] in [np.float, np.float16, np.float32, np.float64]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: fun_float2str(x, intlen, declen))
        elif fdinfo[fs] in [np.int, np.int8, np.int16, np.int32, np.int64]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: fun_int2str(x, 6))
        elif fdinfo[fs] in [str]:
            df[fs+'_f'] = dfsource[fs].apply(lambda x: x.rjust(strlen))
    df.sort_index(axis=1)
    return df


def fun_round45_dep(v, i=0):
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


def fun_round45i(v, dec=0):
    u = int(v * 10**dec*10)
    return (int(u/10) + (1 if v > 0 else -1))/10**dec if (abs(u) % 10 >= 5) else int(u/10)/10**dec


def plot_norm(mean=60, std=15, start=20, end=100, size=1000):
    plt.plot([x for x in np.linspace(start, end, size)],
             [stats.norm.pdf((x - mean) / std) for x in np.linspace(start, end, size)])
    for x in range(start, end+10, 10):
        plt.plot([x, x], [0, stats.norm.pdf((x-mean)/std)], '--')
        plt.text(x, -0.015, '{:.4f}'.format(stats.norm.cdf((x-mean)/std)))
    plt.plot([start, end], [0, 0], '--')
    plt.xlabel('std={:.4f}'.format(std))
    return


# 正态分布测试
def test_norm(data):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(data) < 50:
        p_value = stats.normaltest(data)[1]
        if p_value < 0.05:
            print("use normaltest")
            print("data are not normal distributed")
            return False
        else:
            print("use normaltest")
            print("data are normal distributed")
            return True

    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(data) < 50:
        p_value = stats.shapiro(data)[1]
        if p_value < 0.05:
            print("use shapiro:")
            print("data are not normal distributed")
            return False
        else:
            print("use shapiro:")
            print("data are normal distributed")
            return True

    if 300 >= len(data) >= 50:
        p_value = lillifors(data)[1]
        if p_value < 0.05:
            print("use lillifors:")
            print("data are not normal distributed")
            return False
        else:
            print("use lillifors:")
            print("data are normal distributed")
            return True

    if len(data) > 300:
        p_value = stats.kstest(data, 'norm')[1]
        if p_value < 0.05:
            print("use kstest:")
            print("data are not normal distributed")
            return False
        else:
            print("use kstest:")
            print("data are normal distributed")
            return True


# 对所有样本组进行正态性检验
def test_list_norm(list_groups):
    result = []
    for gi, group in enumerate(list_groups):
        # 正态性检验
        status = test_norm(group)
        if status is False:
            print('the {}-th var is not normal var'.format(gi))
        else:
            print('the {}-th var is normal var'.format(gi))
        result.append(status)
    return result


def read_large_csv(f, display=False):
    if f is str:
        if not os.path.isfile(f):
            if display:
                print('file:{} not found!'.format(f))
                return None
    else:
        if display:
            print('not valid file name:{}'.format(f))
        return None

    reader = pd.read_csv(f, sep=',', iterator=True)
    loop = True
    chunk_size = 100000
    chunks = []
    start = time.time()
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    if display:
        print('use time:{}'.format(time.time()-start))
    return df

def ex_set_place(df: pd.DataFrame, field_list=(),
                 ascending=False, rand_ascending=False,
                 display=False):
    if display:
        print('calculating place...')
    dt = copy.deepcopy(df)
    dt_len = len(dt)
    for fs in field_list:
        if display:
            print('calculate field:{}'.format(fs))

        rand_list = [x for x in range(1, dt_len+1)]
        random.shuffle(rand_list)
        dt.loc[:, fs+'_rand'] = rand_list
        dt = dt.sort_values([fs, fs+'_rand'], ascending=ascending)
        # random is random, no for order
        if not rand_ascending:
            dt[fs+'_rand'] = dt[fs+'_rand'].apply(lambda x: dt_len - x + 1)

        pltemp = []
        last_sv = dt.head(1)[fs].values[0]
        last_pl = 1
        for fi, svi in enumerate(dt[fs].values):
            if svi == last_sv:
                pltemp.append(last_pl)
            else:
                pltemp.append(fi+1)
                last_pl = fi + 1
            last_sv = svi
        dt.loc[:, fs+'_place'] = pltemp
        dt.loc[:, fs+'_place_exact'] = [x+1 for x in range(dt_len)]  # pltemp_exact with random order

    return dt
