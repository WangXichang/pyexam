# -*- coding: utf-8 -*-
# version 2017-09-16
# 2017-11-18


import time
import os
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import numpy as np
import copy
import random
from scipy import stats
from statsmodels.stats.diagnostic import lillifors


def help_doc():
    print("""
    types in this modul:
    fun_ : universal functions
        round45i(v, dec)
        float2str(v, intlen, declen)
        int2str(v, dec)
        round45_dep(v, dec)
    exp_... : create some exmples dataset
        norm_norm_data(mean, std, maxvalue, minvalue, size)
        norm_table(size, std, mean, stdnum)
    report_... : describe some info to dataset
           describe(df, dec)
    exfun_... : some functions for examination data processing
          set_place(df, with_zero, )
    """)


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
def report_describe(df, decimal=4):
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

    pr = [[stats.pearsonr(df[x], df[y])[0]
           for x in df.columns] for y in df.columns]
    sd = stats.describe(df)
    cv = df.cov()
    print('\trecords: ', sd.nobs)
    print('\tpearson recorrelation:')
    for i in range(len(df.columns)):
        print('\t', fun_list2str(pr[i], 4))
    print('\tcovariance matrix:')
    for j in range(len(cv)):
        print('\t', fun_list2str(cv.iloc[j, :], 4))
    print('\tmin : ', fun_list2str(sd.minmax[0], 4))
    print('\tmax : ', fun_list2str(sd.minmax[1], 4))
    print('\tmean: ', fun_list2str(sd.mean, decimal))
    print('\tvar : ', fun_list2str(sd.variance, decimal))
    print('\tstd : ', fun_list2sqrt2str(sd.variance, decimal))
    print('\tskewness: ', fun_list2str(sd.skewness, decimal))
    print('\tkurtosis: ', fun_list2str(sd.kurtosis, decimal))
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


def fun_float2str(v, intlen, declen):
    v = fun_round45i(v, declen)
    fs = '{:'+str(intlen+declen+1)+'.'+str(declen)+'f}'
    return fs.format(v)


def fun_int2str(v, decimal):
    return ('{:' + str(decimal) + 'd}').format(v)


def fun_df_digit2str(dfsource, intlen=2, declen=4, strlen=8):
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


def plot_dist_norm(mean=60, std=15, start=20, end=100, size=1000):
    plt.plot([x for x in np.linspace(start, end, size)],
             [stats.norm.pdf((x - mean) / std) for x in np.linspace(start, end, size)])
    for x in range(start, end+10, 10):
        plt.plot([x, x], [0, stats.norm.pdf((x-mean)/std)], '--')
        plt.text(x, -0.015, '{:.4f}'.format(stats.norm.cdf((x-mean)/std)))
    plt.plot([start, end], [0, 0], '--')
    plt.xlabel('std={:.4f}'.format(std))
    return


# 正态分布测试
def fun_test_norm(data, p_value=0.05):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(data) < 50:
        p_value0 = stats.normaltest(data)[1]
        if p_value0 < p_value:
            print("use normaltest")
            print("data are not normal distributed: p={}<{}".format(p_value0, p_value))
            return False
        else:
            print("use normaltest")
            print("data are normal distributed")
            return True

    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(data) < 50:
        p_value0 = stats.shapiro(data)[1]
        print("use shapiro: p_value={}".format(p_value0))
        if p_value0 < p_value:
            print("data are not normal distributed: p={} < {}".format(p_value0, p_value))
            return False
        else:
            print("data are normal distributed: p={} > {}".format(p_value0, p_value))
            return True

    if 300 >= len(data) >= 50:
        p_value0 = lillifors(data)[1]
        print("use lilifors: p_value={}".format(p_value0))
        if p_value0 < p_value:
            print("data are not normal distributed: p={} < {}".format(p_value0, p_value))
            return False
        else:
            print("data are normal distributed:: p={} > {}".format(p_value0, p_value))
            return True

    if len(data) > 300:
        p_value0 = stats.kstest(data, 'norm')[1]
        print("use kstest: p_value={}".format(p_value))
        if p_value0 < p_value:
            print("data are not normal distributed: p={} < {}".format(p_value0, p_value))
            return False
        else:
            print("data are normal distributed: p={} > {}".format(p_value0, p_value))
            return True


# 对所有样本组进行正态性检验
def fun_test_norm_dataset_list(dataset_list, p_value=0.01):
    result = []
    for gi, dataset in enumerate(dataset_list):
        # 正态性检验
        status = fun_test_norm(dataset, p_value)
        if status is False:
            print('the {}-th is not normal var'.format(gi))
        else:
            print('the {}-th is normal var'.format(gi))
        result.append(status)
    return result


def fun_read_large_csv(file_name, display=False):
    if file_name is str:
        if not os.path.isfile(file_name):
            if display:
                print('file:{} not found!'.format(file_name))
                return None
    else:
        if display:
            print('not valid file name:{}'.format(file_name))
        return None

    reader = pd.read_csv(file_name, sep=',', iterator=True)
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


def fun_ranking(df: pd.DataFrame,
                ranking_field_list=(),
                ranking_suffix_score_field_list=None,
                ranking_rand_field=None,
                score_ascending=False,
                rand_ascending=True,
                score_digits = 3,
                display=False):
    """
    calculate place by some score field
    add score after decimal dot by using suffix fields  in suffix_field_list in order
    place order is set by ascending = False/True
    same place is reset by rand number with rand_ascending=False/True
    :param df: input dataframe
    :param ranking_field_list: set place acoording to the fields  in field_list
    :param ranking_suffix_score_field_list: composite some score fields as suffix of the place field used to sort
    :param ranking_rand_field: rand number used for sorting when order is same at field_list
    :param score_ascending:  place order by score field
    :param rand_ascending: order by random number
    :param score_digits: length of the score, that is int digits
    :param display: display some messages in running procedure
    :return: output dataframe
    """

    if display:
        print('calculating place...')
    dt = copy.deepcopy(df)
    dt_len = len(dt)
    for fs in ranking_field_list:
        if display:
            print('calculate field:{}'.format(fs))
        if len(ranking_suffix_score_field_list) > 0:
            if fs+'_suff' in dt.columns:
                dt.drop(fs+'_suff')
            format_str = '{:0'+str(score_digits)+'d}'   # {:03d}
            dt.loc[:, fs+'_suff'] = [format_str.format(int(x)) for x in dt[fs]]
            for fsf in ranking_suffix_score_field_list:
                dt.loc[:, fs+'_suff'] = dt[fs+'_suff'] + df[fsf].apply(lambda x: format_str.format(int(x)))
        if ranking_rand_field in dt.columns:
            f_rand = ranking_rand_field
        elif ranking_rand_field is None:
            rand_list = [x for x in range(1, dt_len+1)]
            random.shuffle(rand_list)
            dt.loc[:, fs+'_rand'] = rand_list
            f_rand = fs+'_rand'

        # use suffix fields
        if ranking_suffix_score_field_list is None:
            dt = dt.sort_values([fs, f_rand], ascending=[score_ascending, rand_ascending])
        else:
            dt = dt.sort_values([fs+'_suff', f_rand],
                                ascending=[score_ascending, rand_ascending])

        if display:
            print('calculate place...')
        pltemp = []
        last_sv = dt.head(1)[fs].values[0]
        last_pl = 1
        place_field = fs if len(ranking_suffix_score_field_list) == 0 else fs + '_suff'
        for fi, svi in enumerate(dt[place_field].values):
            if svi == last_sv:
                pltemp.append(last_pl)
            else:
                pltemp.append(fi+1)
                last_pl = fi + 1
            last_sv = svi
        dt.loc[:, fs+'_place'] = pltemp
        dt.loc[:, fs+'_place_exact'] = [x+1 for x in range(dt_len)]  # pltemp_exact with random order

    return dt


def report_mean_median_mode(df, field_list, with_zero=False, display=True):
    from scipy.stats import mode
    from numpy import mean, median
    stname = ('mean', 'median', 'mode')
    for field in field_list:
        dft = df[df[field] > 0][field] if not with_zero else df[field]
        st = (mean(dft),
              median(dft),
              mode(dft)[0][0],
              mode(dft)[1][0])
        print('field:{}(with {} zero) '
              'mean={:.2f}, median={:.2f}, mode={} modecount={}'.
              format(field, 'no' if not with_zero else 'all',
                     st[0], st[1], st[2], st[3]))
        if display:
            plt.figure()
            sbn.distplot(dft)
            for i in range(3):
                plt.plot([st[i], st[i]], [0, 1], '--')
                plt.text(st[i], 0.001*(i+1), stname[i])


def fun_dec2bin(x, max_digit=100):
    if x > 1:
        x_int = bin(int(x)).replace('0b', '')
    else:
        x_int = ''
    x -= int(x)
    bins = []
    while max_digit:
        x *= 2
        if x >= 1:
            bins.append('1')
            x -= int(x)
        else:
            bins.append('0')
        max_digit -= 1
    return '{}.{}'.format(x_int, ''.join(bins)) \
        if x_int else '0.{}'.format(''.join(bins))


def fun_bin2dec(b: str):
    b_int = 0
    if '.' in b:
        bi = b[:b.find('.')]
        bd = b[b.find('.')+1:]
        if len(bi) > 0:
            b_int = int(bi, 2)
    else:
        return int(b, 2)
    from decimal import Decimal as Dc
    d = Dc('0')
    for i, x in enumerate(bd):
        d += Dc(2**(-i-1) * int(x))
    return Dc(b_int) + d


def fun_bin4save2dec(f: float, ndigit=55):
    bin4save = uf_dec2bin(f).rstrip('0')
    bin4save = bin4save[bin4save.find('.')+1:]
    ndigit = len(bin4save)
    print(bin4save, ndigit)
    from decimal import Decimal, getcontext
    getcontext().prec = ndigit
    dv = Decimal('0')
    for i, b in enumerate(bin4save):
        dv = dv + Decimal(b) / Decimal(2**(i+1))
    return dv


def fun_decimal_exp(x):
    """Return e raised to the power of x.  Result type matches input type.

    >>> print(exp(Decimal(1)))
    2.718281828459045235360287471
    >>> print(exp(Decimal(2)))
    7.389056098930650227230427461
    >>> print(exp(2.0))
    7.38905609893
    >>> print(exp(2+0j))
    (7.38905609893+0j)

    """
    from decimal import getcontext
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x
        s += num / fact
    getcontext().prec -= 2
    return +s


def fun_decimal_pi(prec=30):
    """
    Compute Pi to the assigned precision
    The default prec = 30, 28digits after decimal point.

    >>> print(pi())
    3.14159265358979323846264338327

    """
    from decimal import Decimal, localcontext
    # tc = getcontext()
    # getcontext().prec = prec + 2  # extra digits for intermediate steps
    with localcontext() as ctx:
        ctx.prec = prec + 2
        # three = Decimal(3)      # substitute "three=3.0" for regular floats
        lasts, t, s, n, na, d, da = 0, Decimal(3), 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    # getcontext().prec -= 2
    return +s


def fun_round45s(v, digits=0):
    __doc__ = '''
    use str and char method
    not valid for digits < 0
    precision is not normal at decimal >16 because of binary representation
    :param number: input float value
    :param digits: places after decimal point
    :return: rounded number with assigned precision
    '''
    vl = str(v).split('.')
    sign = -1 if v < 0 else 1
    if len(vl) == 2:
        if len(vl[1]) > digits:
            if vl[1][digits] >= '5':
                if digits > 0:
                    vp = (eval(vl[1][:digits]) + 1) / 10 ** digits * sign
                    return int(v)+vp
                else:
                    return int(v)+sign
            else:
                if digits > 0:
                    return int(v) + eval(vl[1][:digits]) / 10 ** digits * sign
                else:
                    return int(v)
        else:
            return v
    return int(v)


def fun_round45i(number, digits=0):
    __doc__ = '''
    use multiple 10 power and int method
    precision is not normal at decimal >16 because of binary representation
    :param number: input float value
    :param digits: places after decimal point
    :return: rounded number with assigned precision
    '''
    u = int(number * 10 ** digits * 10)
    return (int(u/10) + (1 if number > 0 else -1)) / 10 ** digits if (abs(u) % 10 >= 5) else int(u / 10) / 10 ** digits


def fun_round45d(v, d, rounding=''):
    __doc__ = '''
    use decimal round method
    precision is not normal beyong decimal precision range,default prec = 28
    :param number: input float value
    :param digits: places after decimal point
    :return: rounded number with assigned precision
    '''
    if 'decimal' not in dir():
        import decimal
    if not rounding:
        rounding = decimal.ROUND_HALF_UP
    vs = str(v)
    if '.' in vs:
        d = d + vs.find('.')
    return float(decimal.Context(prec=d, rounding=decimal.ROUND_HALF_UP).create_decimal(str(v)))


def test_round45(fun, test_num=1000):
    v = 123.275
    st = time.time()
    for _ in range(test_num):
        fun(v, 2)
    print(format((time.time()-st)/test_num, '.30f'))
    return
