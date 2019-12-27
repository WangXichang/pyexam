# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
from stm import modelsetin as msetin, main as main


def plot_models(font_size=12):
    _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'ss7', 'hn900']
    ms_dict = dict()
    for _name in _names:
        ms_dict.update({_name: get_model_describe(name=_name)})

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


def show_models():
    for k in msetin.Models:
        v = msetin.Models[k]
        print('{:<20s} {},  {} '.format(k, v.type, v.desc))
        print('{:<20s} {}'.format(' ', v.ratio))
        print('{:<20s} {}'.format('', v.section))


def get_model_describe(name='shandong'):
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


def check_model(
                model_name=None,
                model_type='plt',
                model_ratio=None,
                model_section=None,
                model_desc=''
                ):
    if model_type not in ['ppt', 'plt']:
        print('error type: valid type must be in {}'.format(model_type, ['ppt', 'plt']))
        return False
    if model_name in msetin.Models:
        print('error name: collision! name {} existed in current mcf.Models_dict!'.format(model_name))
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
    if model_desc is not str:
        print('error desc: model desc(ription) must be str, but real type={}'.format(type(model_desc)))
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


def get_norm_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000, decimal=6):
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
                                  min(round(x, decimal) if decimal > 0 else int(round(x, decimal)),
                                      maxvalue))
                              for x in np.random.normal(mean, std, size)]})
    return df


# test dataset
class TestData:
    def __init__(self, mean=60, std=18, size=100000, max_value=100, min_value=0):
        self.df = None
        self.df_mean = mean
        self.df_max = max_value
        self.df_min = min_value
        self.df_std = std
        self.df_size = size
        self.dist = 'norm'
        self.__make_data()

    def __make_data(self):
        self.df = pd.DataFrame({
            'pid': [str(x).zfill(7) for x in range(1, self.df_size+1)],
            'km1': self.get_score(),
            'km2': self.get_score(),
        })

    def get_score(self):
        print('create score...')
        norm_list = None
        if self.dist == 'norm':
            norm_list = sts.norm.rvs(loc=self.df_mean, scale=self.df_std, size=self.df_size)
            norm_list[np.where(norm_list>self.df_max)] = self.df_max
            norm_list[np.where(norm_list<self.df_min)] = self.df_min
            norm_list = norm_list.astype(np.int)
        return norm_list

    def __call__(self):
        return self.df


def test_stm_with_stat_data(
        name='shandong',
        mode_ratio_cumu='no',
        mode_ratio_prox='upper_min',
        score_max=100,
        score_min=0,
        data_size=1000,
        data_no=1
        ):

    if name.lower() not in msetin.Models.key():
        print('Invalid model name:{}! \ncorrect model name in: [{}]'.
              format(name, ','.join(msetin.Models.key())))
        return None

    # create data set
    print('create test dataset...')

    # --- normal data set
    norm_data1 = [sts.norm.rvs() for _ in range(data_size)]
    norm_data1 = [-4 if x < -4 else (4 if x > 4 else x) for x in norm_data1]
    norm_data1 = [int(x * (score_max - score_min) / 8 + (score_max + score_min) / 2) for x in norm_data1]

    # --- discrete data set
    norm_data2 = []
    for x in range(score_min, score_max, 5):
        if x < (score_min+score_max)/2:
            norm_data2 += [x] * (x % 3)
        else:
            norm_data2 += [x] * (100-x+2)

    # --- triangle data set
    norm_data3 = []
    for x in range(0, score_max+1):
        if x < (score_min+score_max)/2:
            norm_data3 += [x]*(2*x+1)
        else:
            norm_data3 += [x]*2*(score_max-x+1)

    # --- triangle data set
    norm_data4 = TestData(mean=60, size=500000)
    norm_data4.df.km1 = norm_data4.df.km1.apply(lambda x: x if x > 35 else int(35+x*0.3))
    norm_data4.df.km1 = norm_data4.df.km1.apply(lambda x: {35: 0, 36: 3, 37: 5}.get(x, 0) if 35<= x < 38 else x)

    test_data = map(lambda d: pd.DataFrame({'kmx': d}), [norm_data1, norm_data2, norm_data3, list(norm_data4.df.km1)])
    test_data = list(test_data)
    dfscore = test_data[data_no-1]

    if name in msetin.Models.keys():
        print('plt model={}'.format(name))
        print('data set size={}, score range from {} to {}'.
              format(data_size, score_min, score_max))
        m = main.run(name=name,
                     df=dfscore, cols=['kmx'],
                     mode_ratio_prox=mode_ratio_prox,
                     mode_ratio_cumu=mode_ratio_cumu
                     )
        return m
