# -*- utf8 -*-

# import numpy as np
import pandas as pd
import importlib as pb
import os
from collections import namedtuple as ntp
import scipy.stats as sts
from stm import main, stmutil as mutl, stmlib2 as mlib, modelext as mext, modelsetin as msetin


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


class TestLvData():

    def __init__(self):
        self.data = None
        self.result = None

    def load_data(self):
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

        self.data = data_cumu

    @mlib.run_timer
    def test(self, data):
        r_dict = dict()
        for num in range(9):
            r = self.test_stm_with_lvdata(data=data['cumu'+str(num)], cols=['wl'], cumu='yes', name='cumu_'+str(num))
            r_dict.update({'cumu'+str(num): r[0]})
        nocumu_names = ['test'+str(i) for i in range(1, 4)] + \
                       ['test4wl', 'test4hx', 'test5wl', 'test5hx', 'test5sw', 'test6hx', 'test7wl18', 'test7wl19', 'test8']
        for num in range(len(nocumu_names)):
            r = self.test_stm_with_lvdata(data=data[nocumu_names[num]],
                                     cols=('wl', 'hx', 'sw'), cumu='no', name=nocumu_names[num])
            r_dict.update({nocumu_names[num]: r[0]})
        self.result =  r_dict

    def test_stm_with_lvdata(self, data=None, cols=('wl', 'hx', 'sw'), cumu='no', name=''):
        cols_real = [f for f in cols if f in data.columns]
        mr = main.run(df=data, cols=cols_real, mode_ratio_cumu=cumu)
        mr.save_report_to_file('f:/mywrite/新高考改革/modelstestdata/testdata/report_'+name+'.txt')
        result = []
        for col in cols_real:
            result.append([col, len(mr.out_data.query(col+'_plt !='+col+'_stand')) == 0])
        self.result = result
        self.result_model = mr


@mlib.run_timer
def test_hainan(mean=60, size=60000, std=16):
    result = dict()
    ResultTuple = ntp('ResultModel', ['data_model_mode_name', 'result_ascending', 'result_descending'])
    # data1
    #    score point mean is bias to right(high), max==100(count==144), 0-4(count==0,0,0,1,1)
    test_data = mutl.TestData(mean=mean, std=std, size=size)
    for j in range(5):
        model_name = 'hainan'+ (str(j+1) if j>0 else '')
        result_name = model_name+ ('300'+str(j+1) if j > 0 else '900')
        ra = main.run(model_name=model_name, df=test_data.df, cols=['km1'], mode_sort_order='ascending')
        rd = main.run(model_name=model_name, df=test_data.df, cols=['km1'], mode_sort_order='descending')
        result[j] = ResultTuple(result_name, ra, rd)
    return result


class TestShandongData():

    def __init__(self):
        self.df16like = pd.read_csv('e:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df16like.wl = self.df16like.wl.apply(lambda x: int(x*10/11))
        self.df16like.sw = self.df16like.sw.apply(lambda x: int(x*10/9))
        self.df16wen = pd.read_csv('e:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df17like = pd.read_csv('e:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df17like.wl = self.df17like.wl.apply(lambda x: int(x*10/11))
        self.df17like.sw = self.df17like.sw.apply(lambda x: int(x*10/9))
        self.df17wen = pd.read_csv('e:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df18like = pd.read_csv('e:/mywrite/newgk/gkdata/18/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df18like.wl = self.df18like.wl.apply(lambda x: int(x*10/11))
        self.df18like.sw = self.df18like.sw.apply(lambda x: int(x*10/9))
        self.df18wen = pd.read_csv('e:/mywrite/newgk/gkdata/18/wenke.csv', sep=',',
                                   usecols=('zz', 'ls', 'dl'))
        self.models_list = []
        self.model = ntp('model', ['name', 'model'])

    def run_stm(self,
                name='shandong',
                year='16',
                kl='wenke',
                mode_ratio_prox='upper_min',
                mode_ratio_cumu='no',
                mode_sort_order='d',
                all='no'
                ):
        pb.reload(main)
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
            m = main.run(
                model_name=name,
                df=dfs[_run[0] + _run[1]],
                cols=list(dfs[_run[0]+_run[1]]),
                mode_ratio_prox=mode_ratio_prox,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_sort_order=mode_sort_order
                )
            self.models_list.append(
                self.model(name + '_' + _run[0] + '_' + _run[1] + '_' + mode_ratio_prox + '_' + mode_ratio_cumu, m))

    def save_report(self):
        for m in self.models_list:
            _root = 'e:/mywrite/newgk/gkdata/report/report_'
            m[1].save_report_to_file(_root + m[0] + '.txt')


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
    norm_data4 = mutl.TestData(mean=60, size=500000)
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
