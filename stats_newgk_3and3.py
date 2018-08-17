# -*- utf-8 -*-

from numpy import std, mean, var
import pandas as pd
import os
import pyex_seg as sg
import importlib as pb
import scipy.stats as stt
import matplotlib.pyplot as plt
import pyex_lib as pl
import pyex_ptt as ptt


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
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: pl.fun_round45i(x, 0))
            elif fs == 'wl':
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: pl.fun_round45i(x / 11 * 10, 0))
            elif fs == 'sw':
                df.loc[:, fs+'n'] = df[fs].apply(lambda x: pl.fun_round45i(x / 9 * 10, 0))
        df.loc[:, 'kl'] = df['ksh'].apply(lambda x: x[4:10])
        df.loc[:, 'zf'] = df['yw']+df['sx']+df['wy'] + \
                          ((df['dln'] + df['lsn'] + df['zzn']) if kl == 'wk' else \
                               (df['wln'] + df['hxn'] + df['swn']))
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

def desc_score_seg_var(df, kl='wk', year='15', minscore=300, maxscore=400, step=50):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl == 'wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    flist = [fs+'n' if fs not in ['yw', 'sx', 'wy'] else fs for fs in flist] + ['zf']
    print('--- {} correlation for {} ---'.format(year, flist))
    result1 = df[df.zf > 0][flist].corr()
    result1 = result1.applymap(lambda x: round(x, 2))
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
        dftemp = dftemp.applymap(lambda x: round(x, 2))
        # dftemp.loc[:, 'seg'] = [str(st)+'-'+str(st+step)]*len(dftemp)
        if result2 is None:
            result2 = dftemp
        else:
            result2.append(dftemp)
        # result1.loc[:, 'year'] = [year]*len(result1)
        # print(result2)
        print(dftemp)
        print('\n')
        dt1 = dict()
        dt1.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        dt2 = dict()
        dt2.update({'year': [year], 'scope': [str(st) + '-' + str(st+step)]})
        for fs in flist:
            if fs in df.columns:
                # dftemp = df[df.zf.apply(lambda x: st <= x <= st+step)]
                # tvar = round(var(dftemp[fs]), 3)
                tvar = round(dftemp.loc[fs, fs], 2)
                dt1.update({fs+'_var': [tvar]})
                if fs != 'zf':
                    tcor = round(stt.pearsonr(dftemp[fs], dftemp['zf'])[0], 2)
                    dt2.update({fs+'_zf': [tcor]})
        dstd = dstd.append(pd.DataFrame(dt1))
        dcor = dcor.append(pd.DataFrame(dt2))
    print('--- {} segment var for {} ---'.format(year, flist))
    print(dstd[['year', 'scope'] + [fs+'_var' for fs in flist]])
    print('\n')
    print('--- {} segment correlation for {} ---'.format(year, flist))
    print(dcor[['year', 'scope'] + [fs+'_zf' for fs in flist if fs != 'zf']])
    return [result1, result2, dstd, dcor]


def plot_pie_subjects_centage():
    plt.rcParams['font.sans-serif'] = ['SimHei']
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


def plot_pie_xk():

    zy_class_name = ['试验班', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                   '12', '13', '51', '52', '53', '54', '55', '56', '57', '58', '59',
                   '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '88']
    pzy_class_name = zy_class_name[0:13]
    xk_name = ['0', '1', '2', '3', '1/2', '1/3']

    zyall = 34210
    pzyall = 24042

    xk_zycount=[19831, 8633, 737, 43, 1803, 3163]
    xk_zyratio=[0.5796843,  0.25235311,  0.02154341,  0.00125694,  0.05270389, 0.09245835]
    pxk_zycount=[10584, 8201, 713., 43., 1593., 2908]
    pxk_zyratio=[0.4402296 ,  0.34111139,  0.02965643,  0.00178854,  0.06625905, 0.120955]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 16})
    plt.figure(u'选科专业统计')
    plt.subplot(121)
    plt.pie(xk_zycount, labels=xk_name, autopct='%1.2f%%')
    plt.title(u'全部选科专业')
    plt.subplot(122)
    plt.pie(pxk_zycount, labels=xk_name, autopct='%1.2f%%')
    plt.title(u'本科选科专业')

    return

def get_xk_zycount(xkfile='d:/work/newgk/gkdata/xk/xk_type_zycount.csv'):
    xk_name = ['xk0', 'xk1', 'xk2', 'xk3', 'xk21', 'xk31']
    xk_label = ['0', '1', '2', '3', '1/2', '1/3']
    xk_subject = ['d', 'h', 'l', 's', 'w', 'z']

    dc = pd.read_csv('d:/work/newgk/gkdata/xk/xk_zyclass_zycount.txt')
    dc = dc.fillna(0)
    zyclass_name = [x for x in dc.zyclass if x not in ('total','ratio')]
    zyclass_name[0] = '00实验基地班'
    zyclass_name[-1] = '88中外合作'
    # print(zyclass_name)

    dzy = pd.read_csv(xkfile, dtype={'zyclass': str})
    dzy = dzy.fillna(0)
    zyfield=list(dzy.columns.values)
    zyfield.remove('zyclass')
    zy_xk_series = dzy[zyfield].sum()
    # print(zy_xk_series)
    # print(len(zy_xk_series))
    # zy_xk_dict = {ind: zy_xk_series[ind] for ind in zy_xk_series.index}

    dzyp = dzy[dzy.zyclass < '50']  # benke
    zy_xk_series_bk = dzyp[zyfield].sum()
    field_dict = {
        'xk1': [x for x in dzy.columns if 'xk1_' in x],
        'xk2': [x for x in dzy.columns if 'xk2_' in x],
        'xk3': [x for x in dzy.columns if 'xk3_' in x],
        'xk21': [x for x in dzy.columns if 'xk21_' in x],
        'xk31': [x for x in dzy.columns if 'xk31_' in x]}
    dtemp = dzy.copy()
    dtemp2 = dzyp.copy()
    for fs in field_dict.keys():
        dtemp.loc[:, fs] = sum(dzy[fd] for fd in field_dict[fs])
        dtemp2.loc[:, fs] = sum(dzyp[fd] for fd in field_dict[fs])
    dzy = dtemp
    dzyp = dtemp2

    # count for xk_type
    xk_count = dzy[xk_name].sum(axis=0)
    zyclass_count = dzy[xk_name].sum(axis=1)
    type_name = dzy.zyclass

    xk_count_bk = dzyp[xk_name].sum()
    zyclass_count_bk = dzyp[xk_name].sum(axis=1)
    type_name_bk = dzyp.zyclass

    xk_count_zk = [x-y for x,y in zip(xk_count, xk_count_bk)]
    zyclass_count_zk = [x-y for x,y in zip(zyclass_count, zyclass_count_bk)]
    # print(xk_count_zk)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 16})
    plt.figure(u'选科专业统计')
    plt.subplot(131)
    plt.pie(xk_count, labels=xk_label, autopct='%1.2f%%')
    plt.title(u'全部选科专业')
    plt.subplot(132)
    plt.pie(xk_count_bk, labels=xk_label, autopct='%1.2f%%')
    plt.title(u'本科选科专业')
    plt.subplot(133)
    plt.pie(xk_count_zk, labels=xk_label, autopct='%1.2f%%')
    plt.title(u'专科选科专业')

    # print(ptt.make_page(dzy[['zyclass']+xk_name], title='all zy count'))
    # print(ptt.make_page(dzyp[['zyclass']+xk_name], title='benke zy count'))
    # print(type_count)

    return  zy_xk_series, zy_xk_series_bk
