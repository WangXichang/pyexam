# -*- utf8 -*-

import pandas as pd
import numpy as np
import stm as stm
import importlib as pb

class TestModelWithGaokaoData():

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
        self.model_dict = dict()

    def run_stm(self,
                name='shandong',
                year='16',
                kl='wenke',
                mode_ratio_approx='upper_min',
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
                df=dfs[_run[0]+_run[1]],
                cols=list(dfs[_run[0]+_run[1]]),
                mode_ratio_loc=mode_ratio_approx,
                mode_ratio_cum=mode_ratio_cumu,
                mode_score_order=mode_score_order
                )
            self.model_dict.update({name+'_'+_run[0]+'_'+_run[1]+'_'+mode_ratio_approx+'_'+mode_ratio_cumu: m})


    def save_report(self):
        for k in self.model_dict:
            _root = 'd:/mywrite/newgk/gkdata/report/report_'
            self.model_dict[k].save_report_to_file(_root + k + '.txt')


def test_stm_with_stat_data(
        name='shandong',
        mode_ratio_cum='no',
        mode_ratio_loc='upper_min',
        score_max=100,
        score_min=0,
        data_size=1000,
        data_no=1
        ):

    if name.lower() not in stm.stm_models_name:
        print('Invalid model name:{}! \ncorrect model name in: [{}]'.
              format(name, ','.join(stm.stm_models_name)))
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
    norm_data4 = TestData(mean=58, size=500000)
    norm_data4.df.km1 = norm_data4.df.km1.apply(lambda x: x if x > 35 else int(35+x*0.3))

    test_data = map(lambda d: pd.DataFrame({'kmx': d}), [norm_data1, norm_data2, norm_data3, list(norm_data4.df.km1)])
    test_data = list(test_data)
    dfscore = test_data[data_no-1]

    if name in stm.plt_models_dict.keys():
        print('plt model={}'.format(name))
        print('data set size={}, score range from {} to {}'.
              format(data_size, score_min, score_max))
        m = stm.run(name=name, df=dfscore, cols=['kmx'],
                mode_ratio_loc=mode_ratio_loc,
                mode_ratio_cum=mode_ratio_cum
                )
        return m

    elif name.lower() == 'zscore':
        m = stm.Zscore()
        m.set_data(dfscore, cols=['kmx'])
        m.set_para(input_score_max=score_max, input_score_min=score_min)
        m.run()
        return m

    elif name.lower() == 'tscore':
        m = stm.Tscore()
        m.set_data(dfscore, cols=['kmx'])
        m.set_para(raw_score_max=score_max,
                   raw_score_min=score_min,
                   t_score_mean=50,
                   t_score_std=10,
                   t_score_stdnum=4)
        m.run()
        return m
    return None


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
