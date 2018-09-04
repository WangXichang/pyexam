# -*- utf8 -*-

import os
import numpy as np
import pandas as pd
import importlib as imp
import pyex_ptt as ptt
import pyex_bzy_exp


def getzyfun():
    zy = getzy()

    def find():
        imp.reload(pyex_bzy_exp)
        pyex_bzy_exp.find(zy)
        return

    return find


def getzy():
    loc_dell = 'd:/work/data/lq/'
    loc_off = 'f:/studies/lqdata/'
    loc_suface = 'c:/users/wangxichang/zydata/'
    loc_lq = 'd:/zy/'
    loc_list = [loc_suface, loc_dell, loc_lq, loc_off]
    
    zy = Finder()
    for p in loc_list:
        if os.path.isfile(p+'td2017bk_sc.csv'):
            zy.set_datapath(p)
            return zy
    
    print('error: no valid path assigned!')
    return None


class Finder:
    def __init__(self):
        self.path = 'f:/studies/lqdata/'
        self.td16bk1 = None
        self.td16bk2 = None
        self.td17bk = None
        self.td16zk = None
        self.td17zk = None
        self.dflq = None
        self.fd2018pt = None
        self.yxinfo = None
        self.yx16 = None
        self.yx17 = None
        self.yx18 = None

    def set_datapath(self, path):
        self.path = path
        self.load_data()

    def load_data(self):
        self.td16bk1 = pd.read_csv(self.path+'td2016pc1_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td16bk1 = self.td16bk1.fillna(0)
        self.td16bk1.astype(dtype={'wkpos': int, 'lkpos': int})

        self.td16bk2 = pd.read_csv(self.path+'td2016pc2_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td16bk2 = self.td16bk2.fillna(0)
        self.td16bk2.astype(dtype={'wkpos': int, 'lkpos': int})

        self.td17bk = pd.read_csv(self.path+'td2017bk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td17bk = self.td17bk.fillna(0)
        self.td17bk.astype(dtype={'wkpos': int, 'lkpos': int, 'xx': str})

        self.td16zk = pd.read_csv(self.path+'td2016zk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td17zk = pd.read_csv(self.path+'td2017zk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)

        fdfs = self.path + 'fd2018pk.csv'
        if os.path.isfile(fdfs):
            tempdf = pd.read_csv(fdfs, skiprows=22)
            tempdf.astype(dtype={'rs': int})
            # print(tempdf.head())
            temparray = np.array([[x for x in tempdf.loc[y: y+10, 'rs']] for y in range(0, len(tempdf), 11)])
            # print(temparray.shape)
            self.fd2018pt = pd.DataFrame({'fd': temparray[:, 0],
                                          'wk': temparray[:, 1], 'wklj': temparray[:, 2],
                                          'lk': temparray[:, 3], 'lklj': temparray[:, 4],
                                          'ty': temparray[:, 5], 'tylj': temparray[:, 6],
                                          'yw': temparray[:, 7], 'ywlj': temparray[:, 8],
                                          'yl': temparray[:, 9], 'yllj': temparray[:, 10],
                                          })

        if os.path.isfile(self.path+'2015pc2lqk.csv'):
            self.dflq = pd.read_csv(self.path+'2015pc2lqk.csv', sep='\t', low_memory=False)

        if os.path.isfile(self.path+'yxinfo.csv'):
            self.yxinfo = pd.read_csv(self.path+'yxinfo.csv', sep=',', index_col=0,
                                      dtype={'yxdm':str, 'ssdm': str, 'yxjblxdm': str, 'zgdm': str,
                                             'sf211': str, 'sf985': str})
            yxinfoc = self.yxinfo[['yxdm', 'ssdm', 'ssmc', 'zgdm', 'zgmc', 'yxjblxdm', 'yxjblxmc', 'sf985', 'sf211']]
            for fs in ['16', '17', '18']:
                fname = self.path+'yxdf20'+fs+'.txt'
                if os.path.isfile(fname):
                    dt = pd.read_csv(fname, sep='\t', index_col=False,
                                     dtype={'yxdm':str, 'ssdm': str, 'dydm': str, 'jhsxdm': str,
                                            'sfmb': str, 'sf985': str, 'yxdh': str})
                    if fs == '18':
                        self.yx18 = dt.copy()
                        # print(self.yx18.head())
                        self.yx18 =pd.merge(self.yx18, yxinfoc, on='yxdm', how='left')
                    elif fs == '17':
                        self.yx17 = dt.copy()
                        self.yx17 =pd.merge(self.yx17, yxinfoc, on='yxdm', how='left')
                    else:
                        self.yx16 = dt.copy()
                        self.yx16 =pd.merge(self.yx16, yxinfoc, on='yxdm', how='left')
                else:
                    print('load fail:{}'.format(fs))

    def findwc(self, score=500, scope=0):
        df = self.fd2018pt
        fdv = df[df.fd.apply(lambda x: score-scope<=x<=score+scope)]
        if len(fdv) > 0:
            print(ptt.make_page(fdv, title=str('focus on '+str(score))))
        else:
            print('not found data for score={}!'.format(score))

    def somexx(self, xxsubstr=('医学',), kl='wk', cc='bk'):
        ffun = closed_filter(xxsubstr)
        if ffun is False:
            return
        # df1, df2, df3 = None, None, None
        if cc == 'bk':
            print('2016p1---')
            df1 = self.td16bk1[self.td16bk1.xx.apply(ffun)][['xx', 'wkpos', 'lkpos']].\
                sort_values(by=('lkpos' if kl == 'lk' else 'wkpos'))
            print(ptt.make_page(df1, '2016p1'))
            print('2016p2---')
            df2 = self.td16bk2[self.td16bk2.xx.apply(ffun)][['xx', 'wkpos', 'lkpos']].\
                sort_values(by='lkpos' if kl == 'lk' else 'wkpos')
            print(ptt.make_page(df2, '2016p2'))
            print('2017bk---')
            df3 = self.td17bk[self.td17bk.xx.apply(ffun)][['xx', 'wkpos', 'lkpos']].\
                sort_values(by='lkpos' if kl == 'lk' else 'wkpos')
            print(ptt.make_page(df3, '2017bk', align={'xx': 'l'}))
        else:
            # print('2016zk---')
            df1 = self.td16zk[self.td16zk.xx.apply(ffun)][['xx', 'wkpos', 'lkpos']].\
                sort_values(by=('lkpos' if kl == 'lk' else 'wkpos'))
            print(ptt.make_page(df1, title='2016zk', align={'xx': 'l'}))
            # print('2017zk---')
            df2 = self.td17zk[self.td17zk.xx.apply(ffun)][['xx', 'wkpos', 'lkpos']].\
                sort_values(by='lkpos' if kl == 'lk' else 'wkpos')
            print(ptt.make_page(df2, title='2017zk', align={'xx': 'l'}))
        return  # df1, df2, df3

    def findxx(self, low, high, filterlist=('',), kl='wk', cc='bk', align=None):
        posfield = 'wkpos' if kl == 'wk' else 'lkpos'
        align = dict() if align is None else align
        if cc == 'bk':
            # print('2016pc1---')
            df1 = self.get_df_from_pos(self.td16bk1, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            # print('2016pc2---')
            df2 = self.get_df_from_pos(self.td16bk2, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            # print('2017---')
            df3 = self.get_df_from_pos(self.td17bk, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            df3.loc[:, 'xxh'] = df3.xx.apply(lambda x: str(x)[0:4])
            if kl == 'lk':
                df1 = df1.rename(columns={'lkjh': 'lkjh16', 'lkpos': 'lkpos16'})
                df2 = df2.rename(columns={'lkjh': 'lkjh16p2', 'lkpos': 'lkpos16p2'})
                outfields = ['xx', 'lkjh', 'lkjh16', 'lkjh16p2', 'lkpos', 'lkpos16', 'lkpos16p2']
            else:
                df1 = df1.rename(columns={'wkjh': 'wkjh16', 'wkpos': 'wkpos16'})
                df2 = df2.rename(columns={'wkjh': 'wkjh16p2', 'wkpos': 'wkpos16p2'})
                outfields = ['xx', 'wkjh', 'wkjh16', 'wkjh16p2', 'wkpos', 'wkpos16', 'wkpos16p2']
            dfmerge = pd.merge(df3, df1, on='xx', how='outer')
            dfmerge = pd.merge(dfmerge, df2, on='xx', how='outer')[outfields]
            dfmerge = dfmerge.fillna('0')
            if kl == 'lk':
                dfmerge = dfmerge.astype(dtype={'lkpos': int, 'lkpos16': int, 'lkpos16p2': int,
                                                'lkjh': int, 'lkjh16': int, 'lkjh16p2': int
                                                }, errors='ignore')
            else:
                dfmerge = dfmerge.astype(dtype={'wkpos': int, 'wkpos16': int, 'wkpos16p2': int,
                                                'wkjh': int, 'wkjh16': int, 'wkjh16p2': int
                                                }, errors='ignore')
        else:
            # print('2016zk---')
            df1 = self.get_df_from_pos(self.td16zk, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            # print('2017zk---')
            df2 = self.get_df_from_pos(self.td17zk, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            if kl == 'lk':
                df1 = df1.rename(columns={'lkjh': 'lkjh16', 'lkpos': 'lkpos16'})
                df2 = df2.rename(columns={'lkjh': 'lkjh17', 'lkpos': 'lkpos17'})
                outfields = ['xx', 'lkjh16', 'lkjh17', 'lkpos16', 'lkpos17']
            else:
                df1 = df1.rename(columns={'wkjh': 'wkjh16', 'wkpos': 'wkpos16'})
                df2 = df2.rename(columns={'wkjh': 'wkjh17', 'wkpos': 'wkpos17'})
                outfields = ['xx', 'wkjh16', 'wkjh17', 'wkpos16', 'wkpos17']
            dfmerge = pd.merge(df2, df1, on='xx', how='outer')[outfields]
            dfmerge = dfmerge.fillna('0')
            if kl == 'lk':
                dfmerge = dfmerge.astype(dtype={'lkpos16': int, 'lkpos17': int,
                                                'lkjh16': int, 'lkjh17': int
                                                }, errors='ignore')
            else:
                dfmerge = dfmerge.astype(dtype={'wkpos16': int, 'wkpos17': int,
                                                'wkjh16': int, 'wkjh17': int
                                                }, errors='ignore')
        print(ptt.make_page(dfmerge, title='16-17zk', align=align))
        return  # dfmerge

    def lookxx(self, yxlist=('',)):
        yxfilterfun = closed_filter(yxlist)
        df = self.yxinfo[self.yxinfo.yxmc.apply(yxfilterfun)]
        print(ptt.make_mpage(df, '/'.join(yxlist)))

    def findzy(self, lowpos=0, highpos=1000000, xxfilterlist=('',), zyfilterlist=('',)):
        # align = dict() if align is None else align
        if self.dflq is None:
            return pd.DataFrame()
        xxfilterfun = closed_filter(xxfilterlist)
        zyfilterfun = closed_filter(zyfilterlist)
        df = self.dflq[self.dflq.YXMC.apply(xxfilterfun) & self.dflq.ZYMC.apply(zyfilterfun) & \
                       (self.dflq.WC >= lowpos) & (self.dflq.WC <= highpos)].\
            groupby(['YXDH', 'ZYDH'])[['WC', 'YXMC', 'ZYMC']].max()
        if len(df) > 0:
            print(ptt.make_page(df.sort_values('WC'), ''.join(zyfilterlist), align={'YXMC': 'l', 'ZYMC': 'l', 'WC': 'r'}))
        else:
            print('no record found in pos {}--{} for xx={} zy={}'.format(lowpos, highpos, xxfilterlist, zyfilterlist))
        return  # df

    @staticmethod
    def get_df_from_pos(df, lowpos, highpos, posfield, filterlist, kl):
        jh = 'wkjh' if kl == 'wk' else 'lkjh'
        filterfun = closed_filter(filterlist)
        return df[['xx', jh, posfield]][(df[posfield] <= highpos) &
                                        (df[posfield] >= lowpos) &
                                        df.xx.apply(filterfun)].sort_values(by=posfield)


def closed_filter(substr_list):
    substr_list = substr_list
    if (not isinstance(substr_list, list)) & (not isinstance(substr_list, tuple)):
        print('filter is not list')
        return False

    def filterfun(x):
        for s in substr_list:
            if s in str(x):
                return True
        return False

    return filterfun


def add_yxinfo(df, dfyx):
    return pd.merge(df, dfyx, on='yxdm')