# -*- utf8 -*-

import pandas as pd
import numpy as np
from pyex import stm as stm
import importlib as pb
import os
from pytools import wrapper
from collections import namedtuple as ntp


# 有关stm测试的问题：
#
#  一. 空值策略问题：
# （1）空值点策略：如果不考虑中间的空值点，则产生的原始分数划分区间不连续，转换公式就会不同。
#               如果不考虑两端的空值点，则高分不会映射满分，0分一般不会缺失，估计不会受到影响。
#
#  二. 海南模型问题:
#     建议使用分段映射方法，增加分数连续性。在高分区和低分区较为显著。
#     建议使用升序搜索方式，可保证高分点映射到900（300）分值。
#   (1) individual ratio mapping method
#     max score = 900(300) at reatio==1.0 for ascending score order
#     but, min score may at 180-200(for 100-900) or 90-100(for 60-300)
#     with descending order, problem occur at max score.
#
#   (2) weight may decrease to 1/3 if common subject score is 900,
#     it is reasonable if common subjects use raw score 150.
#


def data_lv():
    file_path = 'f:/mywrite/新高考改革/modelstestdata/testdata/'
    cumu_file_list = [file_path+'cumu/'+str(p)+'/out18wl_stand.csv' for p in range(1, 10)]
    nocumu_file_dict = dict()
    data_cumu = dict()
    data_nocumu = dict()
    for i, fname in enumerate(cumu_file_list):
        if os.path.isfile(fname):
            print('load: '+ fname)
        else:
            print('fail:' + fname)
            continue
        with open(fname) as f:
            data_cumu.update({'cumu'+str(i): pd.read_csv(f)})
    nocumu_file_dict.update({'test1': file_path+'nocumu/test1/结果/out2018lz_stand.csv'})
    nocumu_file_dict.update({'test2': file_path+'nocumu/test2/结果/out_stand.csv'})
    nocumu_file_dict.update({'test3': file_path+'nocumu/test3/结果/out2018wl_stand.csv'})

    nocumu_file_dict.update({'test4hx': file_path+'nocumu/test4/结果/out18hx_stand.csv'})
    # test4hx-- find error: seek 0.5 at 0.50000, lv-wang: at 0.509924
    nocumu_file_dict.update({'test4wl': file_path+'nocumu/test4/结果/out18wl_stand.csv'})

    nocumu_file_dict.update({'test5sw': file_path+'nocumu/test5/结果/out18sw_stand.csv'})
    # test5sw-- zero-max: [0, 0] => [21, 30]
    nocumu_file_dict.update({'test5hx': file_path+'nocumu/test5/结果/out18hx_stand.csv'})
    nocumu_file_dict.update({'test5wl': file_path+'nocumu/test5/结果/out18wl_stand.csv'})

    nocumu_file_dict.update({'test6hx': file_path+'nocumu/test6/结果/out2018hx_stand.csv'})

    nocumu_file_dict.update({'test7wl18': file_path+'nocumu/test7/结果/out18wl_stand.csv'})
    nocumu_file_dict.update({'test7wl19': file_path+'nocumu/test7/结果/out19wl_stand.csv'})

    nocumu_file_dict.update({'test8': file_path+'nocumu/test8/out18wl_stand.csv'})

    for dname in nocumu_file_dict:
        if os.path.isfile(nocumu_file_dict[dname]):
            print('load: '+dname)
        else:
            print('fail: '+nocumu_file_dict[dname])
            continue
        with open(nocumu_file_dict[dname]) as f_out:
            data_nocumu.update({dname: pd.read_csv(f_out)})

    # merge test4
    # data_nocumu.update({'test4': pd.merge(data_nocumu['test4wl'], data_nocumu['test4hx'], on='ksh')})
    # data_nocumu.pop('test4wl')
    # data_nocumu.pop('test4hx')
    #
    # # merge test5
    # df = None
    # for sub in ['sw', 'hx', 'wl']:
    #     if sub == 'sw':
    #         df = data_nocumu['test5wl'].__deepcopy__()
    #     else:
    #         df = pd.merge(df, data_nocumu['test5'+sub], on='ksh')
    #     data_nocumu.pop('test5'+sub)
    # data_nocumu.update({'test5': df})

    data_cumu.update(data_nocumu)

    return data_cumu


@wrapper.time_disper
def test_lv(data):
    r_dict = dict()
    for num in range(9):
        r = test_stm_with_lvdata(data=data['cumu'+str(num)], cols=['wl'], cumu='yes', name='cumu_'+str(num))
        r_dict.update({'cumu'+str(num): r[0]})
    nocumu_names = ['test'+str(i) for i in range(1, 4)] + \
                   ['test4wl', 'test4hx', 'test5wl', 'test5hx', 'test5sw', 'test6hx', 'test7wl18', 'test7wl19', 'test8']
    for num in range(len(nocumu_names)):
        r = test_stm_with_lvdata(data=data[nocumu_names[num]],
                                 cols=('wl', 'hx', 'sw'), cumu='no', name=nocumu_names[num])
        r_dict.update({nocumu_names[num]: r[0]})
    return r_dict


def test_stm_with_lvdata(data=None, cols=('wl', 'hx', 'sw'), cumu='no', name=''):
    cols_real = [f for f in cols if f in data.columns]
    mr = stm.run(data=data, cols=cols_real, mode_ratio_cumu=cumu)
    mr.save_report_to_file('f:/mywrite/新高考改革/modelstestdata/testdata/report_'+name+'.txt')
    result = []
    for col in cols_real:
        result.append([col, len(mr.out_data.query(col+'_plt !='+col+'_stand')) == 0])
    return result, mr


def test_hainan(num=1):
    if num == 1:
        # data1
        #    score point mean is bias to right(high), max==100(count==144), 0-4(count==0,0,0,1,1)
        test_data1 = TestData(mean=60, std=14, size=60000)
        # use model100-900
        # score_order=='ascending', out_score_min==277, max==900, second_max=784
        #              'descending', out_score_max==784, second_min==101, 110, 123
        ht1a = stm.run(name='hainan', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht1d = stm.run(name='hainan', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        # use model60-300
        # score_order=='ascending', out_score_min==
        ht2a = stm.run(name='hainan2', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht2d = stm.run(name='hainan2', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')

    if num == 2:
        # data2
        test_data1 = TestData(mean=50, std=14, size=60000)
        # use model100-900
        # score_order=='ascending', out_score_min==150(raw==0, count==12), max==900(count==11), second_max=856(count==6)
        #              'descending', out_score_max==861(count==9), min=100(raw=0, count==7), second_min==132,143 ,158
        ht1a = stm.run(name='hainan', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht1d = stm.run(name='hainan', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        # use model60-300
        # score_order=='ascending', out_score_min==69,73    max==300(100, 9), second_max==288(99, 5)
        #              'descending', out_score_max==288, second_min==60, 69, 73
        ht2a = stm.run(name='hainan2', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht2d = stm.run(name='hainan2', data=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        return ht1a, ht1d, ht2a, ht2d


class TestStmWithSdData():

    def __init__(self):
        self.df16like = pd.read_csv('d:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df16like.wl = self.df16like.wl.apply(lambda x: int(x*10/11))
        self.df16like.sw = self.df16like.sw.apply(lambda x: int(x*10/9))
        self.df16wen = pd.read_csv('d:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df17like = pd.read_csv('d:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df17like.wl = self.df17like.wl.apply(lambda x: int(x*10/11))
        self.df17like.sw = self.df17like.sw.apply(lambda x: int(x*10/9))
        self.df17wen = pd.read_csv('d:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df18like = pd.read_csv('d:/mywrite/newgk/gkdata/18/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df18like.wl = self.df18like.wl.apply(lambda x: int(x*10/11))
        self.df18like.sw = self.df18like.sw.apply(lambda x: int(x*10/9))
        self.df18wen = pd.read_csv('d:/mywrite/newgk/gkdata/18/wenke.csv', sep=',',
                                   usecols=('zz', 'ls', 'dl'))
        self.models_list = []
        self.model = ntp('model', ['name', 'model'])

    def run_stm(self,
                name='shandong',
                year='16',
                kl='wenke',
                mode_ratio_seek='upper_min',
                mode_ratio_cumu='no',
                mode_score_order='d',
                all='no'
                ):
        pb.reload(stm)
        dfs = {
            '16like': self.df16like,
            '16wenke': self.df16wen,
            '17like': self.df17like,
            '17wenke': self.df17wen,
            '18like': self.df18like,
            '18wenke': self.df18wen}
        if all == 'no':
            _all = [(year, kl)]
        else:
            _all = [(s[:2], s[2:]) for s in dfs.keys()]
        for _run in _all:
            m = stm.run(
                name=name,
                data=dfs[_run[0]+_run[1]],
                cols=list(dfs[_run[0]+_run[1]]),
                mode_ratio_seek=mode_ratio_seek,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_score_order=mode_score_order
                )
            self.models_list.append(
                self.model(name + '_' + _run[0] + '_' + _run[1] + '_' + mode_ratio_seek + '_' + mode_ratio_cumu, m))

    def save_report(self):
        for m in self.models_list:
            _root = 'd:/mywrite/newgk/gkdata/report/report_'
            m[1].save_report_to_file(_root + m[0] + '.txt')


def test_stm_with_stat_data(
        name='shandong',
        mode_ratio_cumu='no',
        mode_ratio_seek='upper_min',
        score_max=100,
        score_min=0,
        data_size=1000,
        data_no=1
        ):

    if name.lower() not in stm.MODELS_NAME_LIST:
        print('Invalid model name:{}! \ncorrect model name in: [{}]'.
              format(name, ','.join(stm.MODELS_NAME_LIST)))
        return None

    # create data set
    print('create test dataset...')

    # --- normal data set
    norm_data1 = [stm.sts.norm.rvs() for _ in range(data_size)]
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

    if name in stm.MODELS_RATIO_SEG_DICT.keys():
        print('plt model={}'.format(name))
        print('data set size={}, score range from {} to {}'.
              format(data_size, score_min, score_max))
        m = stm.run(name=name,
                    data=dfscore, cols=['kmx'],
                    mode_ratio_seek=mode_ratio_seek,
                    mode_ratio_cumu=mode_ratio_cumu
                    )
        return m


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
            'no': [str(x).zfill(7) for x in range(1, self.df_size+1)],
            'km1': self.get_score(),
            'km2': self.get_score(),
        })

    def get_score(self):
        print('create score...')
        norm_list = None
        if self.dist == 'norm':
            norm_list = stm.sts.norm.rvs(loc=self.df_mean, scale=self.df_std, size=self.df_size)
            norm_list[np.where(norm_list>self.df_max)] = self.df_max
            norm_list[np.where(norm_list<self.df_min)] = self.df_min
            norm_list = norm_list.astype(np.int)
        return norm_list
