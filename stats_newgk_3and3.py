# -*- utf-8 -*-

from numpy import std, mean, var
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
        flist = ['dl', 'ls', 'zz'] if kl == 'wk' else ['wl', 'hx', 'sw']
        df = datawk[yeari] if kl == 'wk' else datalk[yeari]
        for fs in flist:
            if fs not in ['wl', 'sw']:
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: round(x))
            elif fs == 'wl':
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: round(x/11*10))
            elif fs == 'sw':
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: round(x/9*10))
        df.loc[:, 'kl'] = df['ksh'].apply(lambda x: x[4:10])
        df = df.drop('ksh', axis=1)
        return df
    return get_cj

def disp(df, kl='wk', minscore=500, maxscore=600):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl=='wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    flist = [fs+'n' for fs in flist]
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

def desc(df, kl='wk', year='15', minscore=300, maxscore=400, step=50):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl == 'wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    flist = [fs+'n' if fs not in ['yw', 'sx', 'wy'] else fs for fs in flist]
    print('--- {} correlation for {} ---'.format(year, flist + ['zf']))
    result1 = df[df.zf > 0][flist + ['zf']].corr()
    result1 = result1.applymap(lambda x: round(x, 4))
    # result1.loc[:, 'year'] = [year]*len(result1)
    print(result1)
    d1 = {'year':[], 'scope': []}
    d1.update({fs+'_var': [] for fs in flist})
    dstd = pd.DataFrame(d1)
    dcor = pd.DataFrame(d1)
    for st in range(minscore, maxscore, step):
        dt1 = dict()
        dt1.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        dt2 = dict()
        dt2.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        for fs in flist:
            if fs in df.columns:
                dftemp = df[(df.zf >= st) & (df.zf <= st+step)]
                tvar = round(var(dftemp[fs]), 2)
                tcor = round(stt.pearsonr(dftemp[fs], dftemp['zf'])[0], 4)
                dt1.update({fs+'_var': [tvar]})
                dt2.update({fs+'_cor': [tcor]})
        dstd = dstd.append(pd.DataFrame(dt1))
        dcor = dcor.append(pd.DataFrame(dt2))
    print('--- segment var for {} ---'.format(flist))
    print(dstd[['year', 'scope'] + [fs+'_var' for fs in flist]])
    print('--- segment std for {} ---'.format(flist))
    print(dcor[['year', 'scope'] + [fs+'_cor' for fs in flist]])
    return [result1, dstd, dcor]
