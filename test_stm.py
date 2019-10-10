# -*- utf8 -*-

import pandas as pd
import numpy as np
import stm as stm
import importlib as pb

class TestModel():

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
                year='17',
                kl='like',
                mode_ratio_approx='upper_min',
                mode_ratio_cumu='no',
                mode_score_order='d'
                ):
        pb.reload(stm)
        dfs = {
            '16like': self.df16like,
            '16wenke': self.df16wen,
            '17like': self.df17like,
            '17wenke': self.df17wen,
            '18like': self.df18like,
            '18wenke': self.df18wen}
        m = stm.run_stm(
            name=name,
            df=dfs[year+kl],
            fs=list(dfs[year+kl]),
            mode_ratio_loc=mode_ratio_approx,
            mode_ratio_cum=mode_ratio_cumu,
            mode_score_order=mode_score_order
            )
        self.model_dict.update({name+'_'+year+'_'+kl+'_'+mode_ratio_approx+'_'+mode_ratio_cumu: m})
