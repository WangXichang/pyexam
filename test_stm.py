# -*- utf8 -*-

import pandas as pd
import numpy as np
import pyex_stm as stm
import importlib as pb

class TestModel():

    def __init__(self):
        self.df17like = pd.read_csv('d:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df17wen = pd.read_csv('d:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df18like = pd.read_csv('d:/mywrite/newgk/gkdata/18/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df18wen = pd.read_csv('d:/mywrite/newgk/gkdata/18/wenke.csv', sep=',',
                                   usecols=('zz', 'ls', 'dl'))
        self.model_dict = dict()

    def run_stm(self,
                name='shandong',
                year='17',
                kl='like',
                mode_ratio_approx='upper_min',
                mode_ratio_cumu='yes',
                mode_score_order='d'
                ):
        pb.reload(stm)
        dfs = {'17like': self.df17like, '17wenke': self.df17wen,
               '18like': self.df18like, '18wenke': self.df18wen}
        m = stm.run_stm(
            name=name,
            df=dfs[year+kl],
            fs=list(dfs[year+kl]),
            mode_ratio_approx=mode_ratio_approx,
            mode_ratio_cumu=mode_ratio_cumu,
            mode_score_order=mode_score_order
            )
        self.model_dict.update({year+'_'+kl+'_'+mode_ratio_approx+'_'+mode_ratio_cumu: m})
