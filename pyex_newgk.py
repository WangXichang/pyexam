# -*- utf-8 -*-

# from numpy import std, mean, var
import pandas as pd
import os
from itertools import combinations as cb
import numpy as np
import importlib as pb
import scipy.stats as stt
import matplotlib.pyplot as plt
import pyex_lib as pl
import pyex_ptt as ptt
import pyex_seg as sg


# constant data
shandong_ratio = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]
shandong_level = [21, 31, 41, 51, 61, 71, 81, 91, 100]

zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
shanghai_ratio = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
beijing_ratio = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
tianjin_ratio = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]


def get_cjfun():
    data_path = ['d:/gkcj1517/', 'f:/studies/lqdata/', 'd:/work/newgk/gkdata/xj1517/']
    datawk = []
    datalk = []
    for fs in ['15', '16', '17']:
        for _path in data_path:
            if not os.path.isfile(_path+'g'+fs+'wk.csv'):
                continue
            datawk.append(pd.read_csv(_path+'g'+fs+'wk.csv', index_col=0, low_memory=False,
                                      dtype={'kl': str}))
            datalk.append(pd.read_csv(_path+'g'+fs+'lk.csv', index_col=0, low_memory=False,
                                      dtype={'kl': str}))
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
        # df.loc[:, 'kl'] = df['ksh'].apply(lambda x: x[4:10])
        df.loc[:, 'zf'] = df['yw']+df['sx']+df['wy'] + \
                          ((df['dln'] + df['lsn'] + df['zzn']) if kl == 'wk' else
                           (df['wln'] + df['hxn'] + df['swn']))
        # if 'ksh' in df.columns:
        #     df = df.drop('ksh', axis=1)
        return df

    return get_cj

def desc_score_segtable_plot(df, kl='wk', minscore=500, maxscore=600):
    flist = ['yw', 'sx', 'wy', 'dl', 'ls', 'zz'] if kl == 'wk' else \
        ['yw', 'sx', 'wy', 'wl', 'hx', 'sw']
    flist = [fs+'n' if fs not in 'yw, sx, wy' else fs for fs in flist]
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


def desc_score_seg_var(df, kl='wk', year='15', minscore=400, maxscore=650, step=50):
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


class Xuanke():

    def __init__(self):
        self.xkfile = 'd:/work/newgk/gkdata/xk/xk_type_zycount.csv'
        self.xk_name = ['xk0', 'xk1', 'xk2', 'xk3', 'xk21', 'xk31']
        self.xk_label = ['0', '1', '2', '3', '1/2', '1/3']
        self.km_pinyin1 = ['d', 'h', 'l', 's', 'w', 'z']
        self.km_ccname = ['地理', '化学', '历史', '生物', '物理', '政治']
        self.xk_km_cb = cb(self.km_pinyin1, 3)
        self.xk_comb_ccname = [''.join([self.km_ccname[self.km_pinyin1.index(s)] for s in k])
                               for k in self.xk_km_cb]

        self.dc = None
        self.dzy = None
        self.dzy_benke = None
        self.zyclass_name = None
        self.xk_comb_dict = {}
        self.xk_comb_dict_benke = {}
        self.df_xk_junshi = None
        self.xk_zy_total = None
        self.xk_zy_total_benke = None

        self.load_data()
        self.load_data_military()

        self.xk_zy_total = sum(self.dzy[['xk0', 'xk1', 'xk2', 'xk3', 'xk21', 'xk31']].sum())
        self.xk_comb_percent = {k: self.xk_comb_dict[k]/self.xk_zy_total
                                for k in self.xk_comb_dict}
        self.xk_comb_df = pd.DataFrame({'xkset': list(self.xk_comb_dict.keys()),
                                        'xkcount': list(self.xk_comb_dict.values()),
                                        'xkpercent': list(self.xk_comb_percent.values())})
        self.xk_comb_df.astype({'xkcount': int})
        self.xk_comb_df.xkpercent = self.xk_comb_df.xkpercent.apply(lambda x: round(100*x, 2))

    def load_data(self):
        self.dc = pd.read_csv('d:/work/newgk/gkdata/xk/xk_zyclass_zycount.txt')
        self.dc = self.dc.fillna(0)
        self.zyclass_name = [x for x in self.dc.zyclass if x not in ('total','ratio')]
        self.zyclass_name[0] = '00实验基地班'
        self.zyclass_name[-1] = '88中外合作'
        self.zyclass_name_bk = self.zyclass_name[0:14]
        self.zyclass_name_bk.append(self.zyclass_name[-1])

        # read zy class type data
        dzy = pd.read_csv(self.xkfile, dtype={'zyclass': str})
        dzy = dzy.fillna(0)
        self.dzy = dzy

        zyfield=list(dzy.columns.values)
        zyfield.remove('zyclass')
        zy_xk_series = dzy[zyfield].sum()
        self.xk_comb_dict = {}
        self.xk_km_cb = cb(self.km_pinyin1, 3)
        for xs in self.xk_km_cb:
            zynum = zy_xk_series['xk0']
            xss = ''.join(xs)
            for t in zy_xk_series.index:
                if '_' not in t:
                    continue
                xktype = t[0:t.find('_')]
                xksubs = t[t.find('_')+1:]
                if xktype in 'xk1,xk2,xk3':
                    # print(xksubs, xss)
                    if xksubs in xss:
                        # print(xss,t,zy_xk_series[t])
                        zynum += zy_xk_series[t]
                elif xktype in 'xk21, xk31':
                    if len(set(xs) & set(xksubs)) > 0:
                        # print(xs, t, zy_xk_series[t])
                        zynum += zy_xk_series[t]
                self.xk_comb_dict.update({xss: zynum})
            # print('km-{} zycount={}'.format(xs, zynum))
        self.xk_comb_centage_list = \
            [(x, pl.uf_round45i(self.xk_comb_dict[x]/34210*100,2)) for x in self.xk_comb_dict]

        dzyp = dzy[dzy.zyclass < '50']  # benke
        self.zy_xk_series_bk = dzyp[zyfield].sum()
        self.field_dict = {
            'xk1': [x for x in dzy.columns if 'xk1_' in x],
            'xk2': [x for x in dzy.columns if 'xk2_' in x],
            'xk3': [x for x in dzy.columns if 'xk3_' in x],
            'xk21': [x for x in dzy.columns if 'xk21_' in x],
            'xk31': [x for x in dzy.columns if 'xk31_' in x]}
        dtemp = dzy.copy()
        dtemp2 = dzyp.copy()
        for fs in self.field_dict.keys():
            dtemp.loc[:, fs] = sum(dzy[fd] for fd in self.field_dict[fs])
            dtemp2.loc[:, fs] = sum(dzyp[fd] for fd in self.field_dict[fs])
        self.dzy = dtemp
        self.dzy_benke = dtemp2
        dzy = dtemp
        dzyp = dtemp2

        # count for xk_type
        xk_name = self.xk_name
        self.xk_count = dzy[xk_name].sum(axis=0)
        self.zyclass_count = dzy[xk_name].sum(axis=1)
        self.type_name = dzy.zyclass

        self.xk_count_bk = dzyp[xk_name].sum()
        self.zyclass_count_bk = dzyp[xk_name].sum(axis=1)
        self.type_name_bk = dzyp.zyclass

        self.xk_count_zk = [x-y for x,y in zip(self.xk_count, self.xk_count_bk)]
        self.zyclass_count_zk = [x-y for x,y in zip(self.zyclass_count, self.zyclass_count_bk)]

    def load_data_military(self):
        self.df_xk_junshi = pd.read_csv('d:/work/newgk/gkdata/xk/xk_junshi2020.csv')

        def get_xktype(xkstr):
            if '不限' in xkstr:
                return 'xk0'
            result = ''
            kmstr = ''
            for km, p in zip(self.km_ccname, self.km_pinyin1):
                if km in xkstr:
                    kmstr += p
            if '或' in xkstr:
                if len(xkstr) == 5:
                    result = 'xk21_'
                elif len(xkstr) == 8:
                    result = 'xk31_'
                else:
                    print('or error:{}'.format(xkstr))
                pass
            elif '并' in xkstr:
                if len(xkstr) == 5:
                    result = 'xk2_'
                elif len(xkstr) == 8:
                    result = 'xk3_'
                else:
                    print('and error:{}'.format(xkstr))
                pass
            else:
                if len(xkstr) == 2:
                    result = 'xk1_'
                else:
                    print('error:{}'.format(xkstr))
            return result+kmstr

        dtemp = self.df_xk_junshi
        dtemp.loc[:, 'zyclass'] = dtemp.zydm.apply(lambda x: str(x)[0:2])
        dtemp.loc[:, 'xk_type'] = dtemp.xkkm.apply(lambda x: get_xktype(x))
        self.df_xk_junshi = dtemp


    def plot_pie(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams.update({'font.size': 16})
        plt.figure(u'选科专业统计')
        plt.subplot(131)
        plt.pie(self.xk_count, labels=self.xk_label, autopct='%1.2f%%')
        plt.title(u'全部选科专业')
        plt.subplot(132)
        plt.pie(self.xk_count_bk, labels=self.xk_label, autopct='%1.2f%%')
        plt.title(u'本科选科专业')
        plt.subplot(133)
        plt.pie(self.xk_count_zk, labels=self.xk_label, autopct='%1.2f%%')
        plt.title(u'专科选科专业')

    def print_xk(self):
        print(ptt.make_page(self.dzy[['zyclass']+self.xk_name], title='all zy count'))
        print(ptt.make_page(self.dzy_benke[['zyclass'] + self.xk_name], title='benke zy count'))

    def ptt_zyclass_xktype(self):
        # benke zyclass-xktype-count
        dt = self.dzy_benke[['zyclass', 'xk0'] + list(self.field_dict.keys())]
        dt = dt.astype({fs: int for fs in list(self.field_dict.keys())+['xk0']})
        dt.zyclass = dt.zyclass.apply(lambda x: self.zyclass_name_bk[int(x[0:2])])
        dt.loc[:, 'xk_sum'] = sum(dt[fs] for fs in ['xk0']+list(self.field_dict.keys()))
        dsum = pd.DataFrame(dt.sum())
        dsum.loc['zyclass'] = 'total'
        dsum = dsum.unstack().unstack()
        # print(dsum)
        dt = dt.append(dsum)

        align_dict = {fs: 'r' for fs in list(self.field_dict.keys())+['xk0']}
        align_dict.update({'zyclass': 'l', 'xk_sum': 'r'})
        print(ptt.make_page(dt, title='xk type count for benke', align=align_dict))

        dtt = pd.DataFrame(self.dzy_benke.sum()).unstack().unstack()
        dtt.zyclass = dtt.zyclass.apply(lambda x: 'total')
        dt2 = pd.concat([self.dzy_benke, dtt])
        dt2.zyclass = dt2.zyclass.apply(lambda x: self.zyclass_name_bk[int(x[0:2])] if x != 'total' else x)
        dt2 = dt2.astype({fs: int for fs in dt2.columns.values if fs != 'zyclass'})
        print(ptt.make_page(dt2, align={fs: 'l' if fs=='zyclass' else 'r' for fs in dt2.columns}))


def xk_stats(xkfile='d:/work/newgk/gkdata/xk/xk_type_zycount.csv',
             plot_pie=False,
             ptt_zycount=False,
             ):
    xk_name = ['xk0', 'xk1', 'xk2', 'xk3', 'xk21', 'xk31']
    xk_label = ['0', '1', '2', '3', '1/2', '1/3']
    xk_subject = ['d', 'h', 'l', 's', 'w', 'z']
    xk_sub_cb = cb(xk_subject, 3)

    dc = pd.read_csv('d:/work/newgk/gkdata/xk/xk_zyclass_zycount.txt')
    dc = dc.fillna(0)
    zyclass_name = [x for x in dc.zyclass if x not in ('total','ratio')]
    zyclass_name[0] = '00实验基地班'
    zyclass_name[-1] = '88中外合作'
    # print(zyclass_name)

    # read zy class type data
    dzy = pd.read_csv(xkfile, dtype={'zyclass': str})
    dzy = dzy.fillna(0)
    zyfield=list(dzy.columns.values)
    zyfield.remove('zyclass')
    zy_xk_series = dzy[zyfield].sum()
    xk_comb_dict = {}
    for xs in xk_sub_cb:
        zynum = zy_xk_series['xk0']
        for t in zy_xk_series.index:
            if '_' not in t:
                continue
            xktype = t[0:t.find('_')]
            xksubs = t[t.find('_')+1:]
            if xktype in 'xk1,xk2,xk3':
                xss = ''.join(xs)
                # print(xksubs, xss)
                if xksubs in xss:
                    print(xss,t,zy_xk_series[t])
                    zynum += zy_xk_series[t]
            elif xktype in 'xk21, xk31':
                if len(set(xs) & set(xksubs)) > 0:
                    print(xs, t, zy_xk_series[t])
                    zynum += zy_xk_series[t]
            xk_comb_dict.update({xs: zynum})
        print('km-{} zycount={}'.format(xs, zynum))

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

    if plot_pie:
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

    if ptt_zycount:
        print(ptt.make_page(dzy[['zyclass']+xk_name], title='all zy count'))
        print(ptt.make_page(dzyp[['zyclass']+xk_name], title='benke zy count'))

    return  zy_xk_series, zy_xk_series_bk


class StmStats(object):

    def __init__(self):
        self.d15l = None
        self.d15w = None
        self.d16l = None
        self.d16w = None
        self.d17l = None
        self.d17w = None

    def load_data(self):
        jfun=get_cjfun()
        self.d15w = jfun()
        self.d15l = jfun('15', 'lk')
        self.d16w = jfun('16')
        self.d16l = jfun('16', 'lk')
        self.d17w = jfun('17')
        self.d17l = jfun('17', 'lk')

    @staticmethod
    def show_mean_std(ms, modelname):
        import pyex_stm as stm
        years = ['15', '16', '17']
        wkkm = ['lsn', 'dln', 'zzn']
        lkkm = ['wln', 'hxn', 'swn']
        r_zj = {}
        field_name = '_plt' if modelname == 'shandong' else '_level_score'
        for y in years:
            # calc wenke data
            df = eval('ms.d'+y+'w')
            rw = stm.test(modelname, df, field_list=wkkm)
            rwdesc = rw.output_data.describe()[[fs+field_name for fs in wkkm]].loc[['mean', 'std']]
            r_zj.update({'wk_'+y: rwdesc})
            print('{}:{}-{} output score result:\n {}'.format(modelname, y, 'lk', rwdesc))
            # calc like data
            df = eval('ms.d'+y+'l')
            rw = stm.test(modelname, df, field_list=lkkm)
            rwdesc = rw.output_data.describe()[[fs+field_name for fs in lkkm]].loc[['mean', 'std']]
            r_zj.update({'lk_'+y: rwdesc})
            print('{}:{}-{} output score result:\n {}'.format(modelname, y, 'lk', rwdesc))

        return r_zj


def report_newgk_mean_std():
    """
    mean = (Sigma: xi * pi)
    std  = (Sigma: xi**2 * pi) - mean**2
    :return:
    """
    def get_mean_std(ratio_list, start=100, step=3):
        mean = sum([(start - step*j)*x/100 for j, x in enumerate(ratio_list)])
        std2 = sum([(start - step*j)**2 * x/100 for j, x in enumerate(ratio_list)])
        return mean, round(np.sqrt(std2 - mean**2), 2)

    stats_dict = {}
    stats_dict.update({'zhejiang': get_mean_std(zhejiang_ratio)})
    stats_dict.update({'shanghai': get_mean_std(shanghai_ratio, start=70)})
    stats_dict.update({'beijing': get_mean_std(beijing_ratio)})
    stats_dict.update({'tianjin': get_mean_std(tianjin_ratio)})
    ver_coeff = {'zhejiang': 100/60,
                 'shanghai': 100/30,
                 'beijing': 100/60,
                 'tianjin': 100/60}
    for k in stats_dict:
        print('{}: mean={:.2f}, std={:.2f}, std100={:.2f}'.
              format(k, stats_dict[k][0], stats_dict[k][1], ver_coeff[k]*stats_dict[k][1]))
