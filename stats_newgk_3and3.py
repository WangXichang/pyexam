# -*- utf-8 -*-

from numpy import std, mean
import pandas as pd
import os
import pyex_seg as sg
import importlib as pb
import scipy.stats as stt

def get_cjfun():
    data_path = ['d:/gkcj1517/', 'f:/studies/lqdata/']
    datawk = []
    datalk = []
    for fs in ['15', '16', '17']:
        for _path in data_path:
            if not os.path.isfile(_path+'g'+fs+'wk.csv'):
                continue
            datawk.append(pd.read_csv(_path+'g'+fs+'wk.csv', index_col=0, low_memory=False,
                                      dtype={'ksh': str}))
            datalk.append(pd.read_csv(_path+'g'+fs+'lk.csv', index_col=0, low_memory=False,
                                      dtype={'ksh': str}))

    def get_cj(year='15', kl='wk'):
        if year in '15-16-17':
            yeari = int(year) - 15
        else:
            yeari = 0
        return datawk[yeari] if kl=='wk' else datalk[yeari]

    return get_cj

def disp(df, kl='wk', minscore=500, maxscore=600):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl=='wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    tdf = df[(df.zf <= maxscore) & (df.zf >= minscore)]
    if 'sg' in dir():
        pb.reload(sg)
    seg = sg.SegTable()
    seg.set_data(tdf, flist)
    seg.set_parameters(segmax=150, segmin=0, segsort='a')
    seg.run()
    dfo=seg.output_data
    dfo[[fs+'_count' for fs in flist]].plot()
    print(tdf[flist].describe())
    return dfo

def desc(df, kl='wk', minscore=300, maxscore=400):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl=='wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    dr = pd.DataFrame({'scope': []}.update({fs+'_std': [] for fs in flist}))
    for fs in flist:
        dr2 = dict()
        dr2.update({'scope': [str(minscore)+'-'+str(maxscore)]})
        if fs in df.columns:
            tstd = std(df[fs][(df.zf>=minscore) & (df.zf<=maxscore)])
            tmean = mean(df[fs][(df.zf>=minscore) & (df.zf<=maxscore)])
            print('{}: std = {}, mean = {}'.format(fs,tstd, tmean))
            dr2.update({fs+'_std': [tstd]})
        dr = dr.append(pd.DataFrame(dr2))
    print(dr)
    return dr
