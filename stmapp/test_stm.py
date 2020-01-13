# -*- utf8 -*-

# import numpy as np
import pandas as pd
import importlib as pb
import os
from collections import namedtuple as ntp
import scipy.stats as sts
from stm import main, stm1, stm2, \
     stmlib as slib, models_sys as mdin
from stmapp import models_util as mutl
import itertools as itl

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
#

# Ratio Match and Section Create Problem:
#   (1) first section: maybe degraded, no lost, not step,
#   (2) last section lost: add (-1, -1)
#   (3) middle section lost:
#       must to do: not share endpoint at section after lost section
#       maybe better: section lost is complex problem, not to deal with anything maybe ok!
#   (4) ratio-deadlock: cannot reach bottom because some ratio gap too large
#         ratio-deadlock example(mean=45, std=10):
#         lower_max deadlock:
#         when use lower_max, may loop at some ratio  point and can not reach bottom
#         at zhejiang model:
#         #    <01> ratio: [def:0.0100  dest:0.0100  match:0.0100] => section_map: raw:[ 72,  62] --> out:[100, 100]
#         #    <02> ratio: [def:0.0300  dest:0.0300  match:0.0280] => section_map: raw:[ 61,  58] --> out:[ 97,  97]
#         #    <03> ratio: [def:0.0600  dest:0.0580  match:0.0560] => section_map: raw:[ 57,  55] --> out:[ 94,  94]
#         #    <04> ratio: [def:0.1000  dest:0.0960  match:0.0880] => section_map: raw:[ 54,  53] --> out:[ 91,  91]
#         #    <05> ratio: [def:0.1500  dest:0.1380  match:0.1330] => section_map: raw:[ 52,  51] --> out:[ 88,  88]
#         #    <06> ratio: [def:0.2100  dest:0.1930  match:0.1900] => section_map: raw:[ 50,  49] --> out:[ 85,  85]
#         #    <07> ratio: [def:0.2800  dest:0.2600  match:0.2380] => section_map: raw:[ 48,  47] --> out:[ 82,  82]
#         #    <08> ratio: [def:0.3600  dest:0.3180  match:0.2980] => section_map: raw:[ 46,  45] --> out:[ 79,  79]
#         #    <09> ratio: [def:0.4300  dest:0.3680  match:0.3410] => section_map: raw:[ 44,  44] --> out:[ 76,  76]
#         #    <10> ratio: [def:0.5000  dest:0.4110  match:0.4110] => section_map: raw:[ 43,  42] --> out:[ 73,  73]
#         #    <11> ratio: [def:0.5700  dest:0.4810  match:0.4480] => section_map: raw:[ 41,  41] --> out:[ 70,  70]
#         #    <12> ratio: [def:0.6400  dest:0.5180  match:0.4830] => section_map: raw:[ 40,  40] --> out:[ 67,  67]
#         #    <13> ratio: [def:0.7100  dest:0.5530  match:0.5360] => section_map: raw:[ 39,  39] --> out:[ 64,  64]
#         #    <14> ratio: [def:0.7800  dest:0.6060  match:0.5770] => section_map: raw:[ 38,  38] --> out:[ 61,  61]
#         #    <15> ratio: [def:0.8400  dest:0.6370  match:0.6260] => section_map: raw:[ 37,  37] --> out:[ 58,  58]
#         #    <16> ratio: [def:0.8900  dest:0.6760  match:0.6570] => section_map: raw:[ 36,  36] --> out:[ 55,  55]
#         #    <17> ratio: [def:0.9300  dest:0.6970  match:0.6570] => section_map: raw:[ 35,  36] --> out:[ 52,  52]
#         #    <18> ratio: [def:0.9600  dest:0.6870  match:0.6570] => section_map: raw:[ 35,  36] --> out:[ 49,  49]
#         #    <19> ratio: [def:0.9800  dest:0.6770  match:0.6570] => section_map: raw:[ 35,  36] --> out:[ 46,  46]
#         #    <20> ratio: [def:0.9900  dest:0.6670  match:0.6570] => section_map: raw:[ 35,  36] --> out:[ 43,  43]
#         #    <21> ratio: [def:1.0000  dest:0.6670  match:0.6570] => section_map: raw:[ 35,  36] --> out:[ 40,  40]
#

def test_all_strategy(df=None, model_name='shandong'):
    # pb.reload(stm2)
    # pb.reload(stm1)
    if df is None:
        df = mutl.TestData(mean=45, std=12, size=1000)()
    print([k for k in mdin.Strategy.keys()])
    ss = [mdin.Strategy[s] for s in mdin.Strategy.keys()]
    sn = [s for s in mdin.Strategy.keys()]
    st = list(itl.product(*ss))
    result = ntp('r', ['df1', 'df2', 'map1', 'map2'])
    r = dict()
    for num, ti in enumerate(st):
        verify = True
        if num != 18:
            continue
        print(num, ti)
        r = main.runm(df=df, cols=['km1'],
                      model_name=model_name,
                      mode_ratio_prox=ti[0],
                      mode_ratio_cumu=ti[1],
                      mode_sort_order=ti[2],
                      mode_section_point_first=ti[3],
                      mode_section_point_start=ti[4],
                      mode_section_point_last=ti[5],
                      mode_section_degraded=ti[6],
                      mode_section_lost=ti[7],
                      verify=verify,
                      )
        if verify:
            if not r[0]:
                return r
    return r


class TestLvData():

    def __init__(self):
        self.data = None
        self.result = None
        self.path = None

    def load_data(self):
        file_path1 = 'f:/mywrite/新高考改革/modelstestdata/testdata/'
        file_path2 = 'e:/mywrite/newgk/lvdata/testdata/'
        if os.path.isdir(file_path1):
            file_path = file_path1
        elif os.path.isdir(file_path2):
            file_path = file_path2
        else:
            print('no valid data path!')
            return
        self.path = file_path

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

    @slib.timer_wrapper
    def test(self, data):
        r_dict = dict()
        for num in range(9):
            r = self.test_stm_with_lvdata(name='cumu'+str(num),
                                          data=data['cumu'+str(num)],
                                          cols=['wl'],
                                          cumu='yes')
            r_dict.update({'cumu'+str(num): r})
        nocumu_names = ['test'+str(i) for i in range(1, 4)] + \
                       ['test4wl', 'test4hx', 'test5wl', 'test5hx', 'test5sw', 'test6hx', 'test7wl18', 'test7wl19', 'test8']
        for num in range(len(nocumu_names)):
            r = self.test_stm_with_lvdata(data=data[nocumu_names[num]],
                                          cols=('wl', 'hx', 'sw'), cumu='no', name=nocumu_names[num])
            r_dict.update({nocumu_names[num]: r})
        self.result = r_dict

    def test_stm_with_lvdata(self, name, data=None, cols=None, cumu=None):
        cols_real = [f for f in cols if f in data.columns]
        print('test {}: {} ... '.format(name, cols))
        mr = main.runm(df=data, cols=cols_real,
                       mode_ratio_cumu=cumu,
                       display=False,
                       logout=True
                       )
        # mr.save_report_doc(self.path + 'report/r2_' + name + '.txt')
        result = []
        for col in cols_real:
            comp = all(mr.outdf[col+'_ts'] == mr.outdf[col+'_stand'])
            result.append([col, comp])
            if not comp:
                print('not equal: name={}, col={}'.format(name, col))
        # self.result = result
        self.result_model = mr
        return result


@slib.timer_wrapper
def test_hainan(mean=60, size=60000, std=16):
    result = dict()
    ResultTuple = ntp('ResultModel', ['data_model_mode_name', 'result_ascending', 'result_descending'])
    # data1
    #    score point mean is bias to right(high), max==100(count==144), 0-4(count==0,0,0,1,1)
    test_data = mutl.TestData(mean=mean, std=std, size=size)
    for j in range(5):
        model_name = 'hainan'+ (str(j+1) if j>0 else '')
        result_name = model_name+ ('300'+str(j+1) if j > 0 else '900')
        ra = main.runm(model_name=model_name, df=test_data.df, cols=['km1'], mode_sort_order='ascending')
        rd = main.runm(model_name=model_name, df=test_data.df, cols=['km1'], mode_sort_order='descending')
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
            m = main.runm(
                model_name=name,
                df=dfs[_run[0] + _run[1]],
                cols=list(dfs[_run[0]+_run[1]]),
                mode_ratio_prox=mode_ratio_prox,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_sort_order=mode_sort_order,
                display=1,
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

    if name.lower() not in mdin.Models.key():
        print('Invalid model name:{}! \ncorrect model name in: [{}]'.
              format(name, ','.join(mdin.Models.key())))
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

    if name in mdin.Models.keys():
        print('plt model={}'.format(name))
        print('data set size={}, score range from {} to {}'.
              format(data_size, score_min, score_max))
        m = main.runm(model_name=name,
                      df=dfscore, cols=['kmx'],
                      mode_ratio_prox=mode_ratio_prox,
                      mode_ratio_cumu=mode_ratio_cumu
                     )
        return m


# class Zscore():
#     """
#     transform raw score to Z-score according to percent position on normal cdf
#     input data: df = raw score dataframe
#     set parameters: stdNum = standard error numbers
#     output data: outdf = result score with raw score field name + '_z'
#     """
#     # HighPrecise = 0.9999999
#     MinError = 0.1 ** 9
#
#     def __init__(self, model_name='zscore'):
#         import array
#
#         # super(Zscore, self).__init__(model_name)
#
#         # input data
#         self.df = None
#         self.cols = None
#
#         # model parameters
#         self.out_score_std_num = 4
#         self.raw_score_max = 100
#         self.raw_score_min = 0
#         self.out_score_decimal = 8
#         self.out_score_point_number = 1000
#         self.norm_table = array.array('d',
#                                       [sts.norm.cdf(-self.out_score_std_num * (1 - 2 * x / (self.out_score_point_number - 1)))
#                                        for x in range(self.out_score_point_number)]
#                                       )
#         # strategies
#         self.mode_ratio_prox = 'near'
#         self.mode_sort_order = 'd'
#
#         # result data
#         self.map_table = None
#         self.outdf = None
#
#     def set_data(self, df=None, cols=None):
#         self.df = df
#         self.cols = cols
#
#     def set_para(self,
#                  std_num=4,
#                  raw_score_defined_min=0,
#                  raw_score_defined_max=100,
#                  mode_ratio_prox='near_max',
#                  mode_sort_order='d',
#                  mode_score_zero='real',
#                  out_decimal=8,
#                  ):
#         self.out_score_std_num = std_num
#         self.raw_score_max = raw_score_defined_max
#         self.raw_score_min = raw_score_defined_min
#         self.mode_ratio_prox = mode_ratio_prox
#         self.mode_sort_order = mode_sort_order
#         self.out_score_decimal = out_decimal
#         self.mode_score_zero=mode_score_zero
#
#     # Zscore run
#     def run(self):
#         print('start run...')
#         st = time.clock()
#         self.outdf = self.df
#         self.map_table = self.get_map_table(
#             self.outdf,
#             self.raw_score_max,
#             self.raw_score_min,
#             self.cols,
#             seg_order=self.mode_sort_order)
#         for col in self.cols:
#             print('calc zscore on field: {}...'.format(col))
#             self.map_table[col+'_zscore'] = self.get_zscore(self.map_table[col+'_percent'])
#             map_dict = {rscore: zscore for rscore, zscore in
#                         zip(self.map_table['seg'], self.map_table[col + '_zscore'])}
#             self.outdf.loc[:, col + '_zscore'] = \
#                 self.outdf[col].apply(lambda x: map_dict.get(x, -999))
#         print('zscore finished with {} consumed'.format(round(time.clock()-st, 2)))
#
#     # new method for uniform algorithm with strategies
#     def get_zscore(self, percent_list):
#         # z_list = [None for _ in percent_list]
#         z_array = array.array('d', range(len(percent_list)))
#         _len = self.out_score_point_number
#         for i, _p in enumerate(percent_list):
#             # do not use mode_ratio_prox
#             pos = bst.bisect(self.norm_table, _p)
#             z_array[i] = 2*(pos - _len/2) / _len * self.out_score_std_num
#         return z_array
#
#     @staticmethod
#     def get_map_table(df, maxscore, minscore, cols, seg_order='a'):
#         seg = slib.SegTable()
#         seg.set_data(df, cols)
#         seg.set_para(segmax=maxscore, segmin=minscore, segsort=seg_order)
#         seg.run()
#         return seg.outdf
#
#     def report(self):
#         if type(self.outdf) == pd.DataFrame:
#             print('output score desc:\n', self.outdf.describe())
#         else:
#             print('output score data is not ready!')
#         print('data fields in raw_score:{}'.format(self.cols))
#         print('para:')
#         print('\tzscore stadard diff numbers:{}'.format(self.out_score_std_num))
#         print('\tmax score in raw score:{}'.format(self.raw_score_max))
#         print('\tmin score in raw score:{}'.format(self.raw_score_min))
#
#     def plot(self, mode='out'):
#         if mode in 'raw,out':
#             super(Zscore, self).plot(mode)
#         else:
#             print('not support this mode!')
#
# # === Zscore model end ===
#
#
# class Tscore():
#     __doc__ = '''
#     T分数是一种标准分常模,平均数为50,标准差为10的分数。
#     即这一词最早由麦柯尔于1939年提出,是为了纪念推孟和桑代克
#     对智力测验,尤其是提出智商这一概念所作出的巨大贡献。
#     通过调整t_score_mean, t_score_std, 也可以进行其它标准分数转换，
#     如100-900分的标准分数转换。
#     本模型使用百分位-累计分布校准的方式计算转换分数。
#     '''
#
#     def __init__(self, model_name='tscore'):
#         # super(Tscore, self).__init__(model_name)
#
#         self.mode_score_paper_max = 100
#         self.mode_score_paper_min = 0
#         self.t_score_std = 10
#         self.t_score_mean = 50
#         self.t_score_stdnum = 4
#
#         self.outdf_decimal = 0
#         self.zscore_decimal = 8
#
#         self.map_table = None
#
#     def set_data(self, df=None, cols=None):
#         self.df = df
#         self.cols = cols
#
#     def set_para(self,
#                  mode_score_defined_min=0,
#                  mode_score_defined_max=100,
#                  t_score_mean=50,
#                  t_score_std=10,
#                  t_score_stdnum=4,
#                  out_decimal=0):
#         self.mode_score_paper_max = mode_score_defined_max
#         self.mode_score_paper_min = mode_score_defined_min
#         self.t_score_mean = t_score_mean
#         self.t_score_std = t_score_std
#         self.t_score_stdnum = t_score_stdnum
#         self.outdf_decimal = out_decimal
#
#     # Tscore
#     def run(self):
#         """get tscore from zscore"""
#         zm = Zscore()
#         zm.set_data(self.df, self.cols)
#         zm.set_para(std_num=self.t_score_stdnum,
#                     raw_score_defined_min=self.mode_score_paper_min,
#                     raw_score_defined_max=self.mode_score_paper_max,
#                     out_decimal=self.zscore_decimal
#                     )
#         zm.run()
#         self.outdf = zm.outdf
#         namelist = self.outdf.columns
#
#         def formula(x):
#             return slib.round45r(x * self.t_score_std + self.t_score_mean, self.outdf_decimal)
#
#         for sf in namelist:
#             if '_zscore' in sf:
#                 new_sf = sf.replace('_zscore', '_tscore')
#                 self.outdf.loc[:, new_sf] = self.outdf[sf].apply(formula)
#         self.map_table = zm.map_table
#
#     def report(self):
#         print('T-score by normal table transform report')
#         print('-' * 50)
#         if type(self.df) == pd.DataFrame:
#             print('raw score desc:')
#             print('    fields:', self.cols)
#             print(self.df[self.cols].describe())
#             print('-'*50)
#         else:
#             print('output score data is not ready!')
#         if type(self.outdf) == pd.DataFrame:
#             out_fields = [f+'_tscore' for f in self.cols]
#             print('T-score desc:')
#             print('    fields:', out_fields)
#             print(self.outdf[out_fields].describe())
#             print('-'*50)
#         else:
#             print('output score data is not ready!')
#         print('data fields in raw_score:{}'.format(self.cols))
#         print('-' * 50)
#         print('para:')
#         print('\tzscore stadard deviation numbers:{}'.format(self.t_score_std))
#         print('\tmax score in raw score:{}'.format(self.mode_score_paper_max))
#         print('\tmin score in raw score:{}'.format(self.mode_score_paper_min))
#         print('-' * 50)
#
#     def plot(self, mode='raw'):
#         super(Tscore, self).plot(mode)
#
#
# class TaiScore():
#     """
#     Grade Score Model used by Taiwan College Admission Test Center
#     top_group = df.sort_values(field,ascending=False).head(int(df.count(0)[field]*0.01))[[field]]
#     high_grade_score = round(top_group[field].mean(), 4)
#     intervals = [minscore, grade_level/grade_level_total_number], ..., [,high_grade]
#     以原始分值切分，形成的分值相当于等距合并，粒度直接增加
#     实质上失去了等级分数的意义
#     本模型仍然存在高分区过度合并问题
#     """
#
#     def __init__(self, model_name='tai'):
#         # super(TaiScore, self).__init__(model_name)
#         self.model_name = 'Taiwan'
#
#         self.grade_num = 15
#         self.mode_score_paper_max = 100
#         self.mode_score_paper_min = 0
#         self.max_ratio = 0.01   # 1%
#         self.df = pd.DataFrame()
#
#         self.grade_no = [x for x in range(self.grade_num+1)]
#         self.map_table = None
#         self.grade_dist_dict = {}  # col: grade_list, from max to min
#         self.outdf = pd.DataFrame()
#
#     def set_data(self, df=pd.DataFrame(), cols=None):
#         if len(df) > 0:
#             self.df = df
#         if isinstance(cols, list) or isinstance(cols, tuple):
#             self.cols = cols
#
#     def set_para(self,
#                  mode_score_paper_max=None,
#                  mode_score_paper_min=None,
#                  grade_num=None,
#                  ):
#         if isinstance(mode_score_paper_max, int):
#             if len(self.cols) > 0:
#                 if mode_score_paper_max >= max([max(self.df[f]) for f in self.cols]):
#                     self.mode_score_paper_max = mode_score_paper_max
#                 else:
#                     print('error: maxscore is too little to transform score!')
#             else:
#                 print('to set col first!')
#         if isinstance(mode_score_paper_min, int):
#             self.mode_score_paper_min = mode_score_paper_min
#         if isinstance(grade_num, int):
#             self.grade_num = grade_num
#         self.grade_no = [x for x in range(self.grade_num+1)]
#
#     def run(self):
#         self.run_create_grade_dist_list()
#         self.run_create_outdf()
#
#     def run_create_grade_dist_list(self):
#         # mode_ratio_prox = 'near'
#         seg = slib.SegTable()
#         seg.set_para(segmax=self.mode_score_paper_max,
#                      segmin=self.mode_score_paper_min,
#                      segsort='d')
#         seg.set_data(self.df,
#                      self.cols)
#         seg.run()
#         self.map_table = seg.outdf
#         for fs in self.cols:
#             lastpercent = 0
#             lastseg = self.mode_score_paper_max
#             for ind, row in self.map_table.iterrows():
#                 curpercent = row[fs + '_percent']
#                 curseg = row['seg']
#                 if row[fs+'_percent'] > self.max_ratio:
#                     if curpercent - self.max_ratio > self.max_ratio - lastpercent:
#                         max_score = lastseg
#                     else:
#                         max_score = curseg
#                     max_point = self.df[self.df[fs] >= max_score][fs].mean()
#                     # print(fs, max_score, curseg, lastseg)
#                     self.grade_dist_dict.update({fs: slib.round45r(max_point / self.grade_num, 8)})
#                     break
#                 lastpercent = curpercent
#                 lastseg = curseg
#
#     def run_create_outdf(self):
#         dt = copy.deepcopy(self.df[self.cols])
#         for fs in self.cols:
#             dt.loc[:, fs+'_grade'] = dt[fs].apply(lambda x: self.run_get_grade_score(fs, x))
#             dt2 = self.map_table
#             dt2.loc[:, fs+'_grade'] = dt2['seg'].apply(lambda x: self.run_get_grade_score(fs, x))
#             self.outdf = dt
#
#     def run_get_grade_score(self, fs, x):
#         if x == 0:
#             return x
#         grade_dist = self.grade_dist_dict[fs]
#         for i in range(self.grade_num):
#             minx = i * grade_dist
#             maxx = (i+1) * grade_dist if i < self.grade_num-1 else self.mode_score_paper_max
#             if minx < x <= maxx:
#                 return i+1
#         return -1
#
#     def plot(self, mode='raw'):
#         pass
#
#     def report(self):
#         print(self.outdf[[f+'_grade' for f in self.cols]].describe())
#
#     def print_map_table(self):
#         print(self.map_table)
