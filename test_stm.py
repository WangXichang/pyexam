# -*- utf8 -*-

import pandas as pd
import numpy as np
import pyex_stm as stm
import importlib as pb

class Score():

    def __init__(self):
        self.df17li = pd.read_csv('d:/mywrite/newgk/gkdata/17/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df17wen = pd.read_csv('d:/mywrite/newgk/gkdata/17/wenke.csv', sep=',',
                                  usecols=('zz', 'ls', 'dl'))
        self.df18li = pd.read_csv('d:/mywrite/newgk/gkdata/18/like.csv', sep=',',
                                  usecols=('wl', 'hx', 'sw'))
        self.df18wen = pd.read_csv('d:/mywrite/newgk/gkdata/18/wenke.csv', sep=',',
                                   usecols=('zz', 'ls', 'dl'))

    def run_stm(self,
                name='shandong',
                year='17',
                kl='like',
                mode_approx='upper_min',
                mode_cumu='yes',
                mode_score_order='d'
                ):
        pb.reload(stm)
        dfs = {'17like': self.df17li, '17wenke': self.df17wen,
               '18like': self.df18li, '18wenke': self.df18wen}
        return stm.run_stm(name=name, df=dfs[year+kl],
                           field_list=list(dfs[year+kl]),
                           mode_approx=mode_approx,
                           mode_cumu=mode_cumu,
                           mode_score_order=mode_score_order
                           )
