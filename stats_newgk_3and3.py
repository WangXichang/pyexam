# -*- utf-8 -*-

from numpy import std, mean, var
import pandas as pd
import os
import pyex_seg as sg
import importlib as pb
import scipy.stats as stt
import matplotlib.pyplot as plt


def get_cjfun():
    data_path = ['d:/gkcj1517/', 'f:/studies/lqdata/', 'd:/work/newgk/gkdata/xj1517/']
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
    # print(datawk, datalk)

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
    flist = [fs+'n' if fs not in ['yw', 'sx', 'wy'] else fs for fs in flist] + ['zf']
    print('--- {} correlation for {} ---'.format(year, flist))
    result1 = df[df.zf > 0][flist].corr()
    result1 = result1.applymap(lambda x: round(x, 4))
    print(result1)
    print('\n')
    result2 = None
    d1 = {'year':[], 'scope': []}
    d1.update({fs+'_var': [] for fs in flist})
    dstd = pd.DataFrame(d1)
    dcor = pd.DataFrame(d1)
    for st in range(minscore, maxscore, step):
        print('--- {} covar for {} ---'.format(year, str(st) + '-' + str(st+step)))
        dftemp = df[df.zf.apply(lambda x: st <= x <= st+step)][flist].cov()
        dftemp = dftemp.applymap(lambda x: round(x, 4))
        dftemp.loc[:, 'seg'] = [str(st)+'-'+str(st+step)]*len(dftemp)
        if result2 is None:
            result2 = dftemp
        else:
            result2.append(dftemp)
        # result1.loc[:, 'year'] = [year]*len(result1)
        print(result2)
        print('\n')
        dt1 = dict()
        dt1.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        dt2 = dict()
        dt2.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        for fs in flist:
            if fs in df.columns:
                dftemp = df[(df.zf >= st) & (df.zf <= st+step)]
                tvar = round(var(dftemp[fs]), 2)
                dt1.update({fs+'_var': [tvar]})
                if fs != 'zf':
                    tcor = round(stt.pearsonr(dftemp[fs], dftemp['zf'])[0], 4)
                    dt2.update({fs+'_zf': [tcor]})
        dstd = dstd.append(pd.DataFrame(dt1))
        dcor = dcor.append(pd.DataFrame(dt2))
    print('--- segment var for {} ---'.format(flist))
    print(dstd[['year', 'scope'] + [fs+'_var' for fs in flist]])
    print('\n')
    print('--- segment std for {} ---'.format(flist))
    print(dcor[['year', 'scope'] + [fs+'_zf' for fs in flist if fs != 'zf']])
    return [result1, result2, dstd, dcor]


def plot_pie():
    plt.subplot(141)
    plt.title('各科目值域总分贡献率')
    plt.pie([150, 150, 150, 100, 100, 100],
            labels=['语文', '数学', '外语', '选科1', '选科2', '选科3'],
            autopct='%1.1f%%')
    plt.subplot(142)
    plt.title('各科目有效值域总分贡献率（山东方案）')
    plt.pie([150, 150, 150, 80, 80, 80, 60],
            labels=['语文', '数学', '外语', '选科1', '选科2', '选科3', '基础分'],
            autopct='%1.1f%%')
    plt.subplot(143)
    plt.title('各科目有效值域总分贡献率（浙江方案）')
    plt.pie([20, 20, 20, 8, 8, 8, 16],
            labels=['语文', '数学', '外语', '选科1', '选科2', '选科3', '基础分'],
            autopct='%1.1f%%')
    plt.subplot(144)
    plt.title('各科目有效值域总分贡献率（上海方案）')
    plt.pie([150, 150, 150, 30, 30, 30, 120],
            labels=['语文', '数学', '外语', '选科1', '选科2', '选科3', '基础分'],
            autopct='%1.1f%%')

