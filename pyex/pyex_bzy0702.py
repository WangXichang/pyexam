# -*- utf -*-

import os
import pandas as pd
from prettytable import PrettyTable as Ptt


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
        self.td16p1 = None
        self.td16p2 = None
        self.td17bk = None
        self.td16zk = None
        self.td17zk = None
        self.dflq = None

    def set_datapath(self, path):
        self.path = path
        self.load_data()

    def load_data(self):
        self.td16p1 = pd.read_csv(self.path+'td2016pc1_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td16p1 = self.td16p1.fillna(0)
        self.td16p1.astype(dtype={'wkpos': int, 'lkpos': int})

        self.td16p2 = pd.read_csv(self.path+'td2016pc2_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td16p2 = self.td16p2.fillna(0)
        self.td16p2.astype(dtype={'wkpos': int, 'lkpos': int})

        self.td17bk = pd.read_csv(self.path+'td2017bk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td17bk = self.td17bk.fillna(0)
        self.td17bk.astype(dtype={'wkpos': int, 'lkpos': int, 'xx': str})

        self.td16zk = pd.read_csv(self.path+'td2016zk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)
        self.td17zk = pd.read_csv(self.path+'td2017zk_sc.csv', sep=',',
                                  dtype={'xx': str}, verbose=True)

        if os.path.isfile(self.path+'2015pc2lqk.csv'):
            self.dflq = pd.read_csv(self.path+'2015pc2lqk.csv', sep='\t', low_memory=False)

    def somexx(self, xxsubstr=('医学',), kl='wk', cc='bk'):
        ffun = closed_filter(xxsubstr)
        if ffun is False:
            return
        df1, df2, df3 = None, None, None
        outfields = ['xx', 'wkpos'] if kl == 'wk' else ['xx', 'lkpos']
        sortfield = 'lkpos' if kl == 'lk' else 'wkpos'
        if cc == 'bk':
            # print('2016p1---')
            df1 = self.td16p1[self.td16p1.xx.apply(ffun)][outfields].\
                  sort_values(by=sortfield).astype({sortfield: int})
            print(make_page(df1, '2016p1'))
            # print('2016p2---')
            df2 = self.td16p2[self.td16p2.xx.apply(ffun)][outfields].\
                  sort_values(by=sortfield).astype({sortfield: int})
            print(make_page(df2, '2016p2'))
            # print('2017bk---')
            df3 = self.td17bk[self.td17bk.xx.apply(ffun)][outfields].\
                  sort_values(by=sortfield).astype({sortfield: int})
            print(make_page(df3, '2017bk', align={'xx': 'l'}))
        else:
            # print('2016zk---')
            df1 = self.td16zk[self.td16zk.xx.apply(ffun)][outfields].\
                  sort_values(by=(sortfield)).astype({sortfield: int})
            print(make_page(df1, title='2016zk', align={'xx': 'l'}))
            # print('2017zk---')
            df2 = self.td17zk[self.td17zk.xx.apply(ffun)][outfields].\
                  sort_values(by=sortfield).astype({sortfield: int})
            print(make_page(df2, title='2017zk', align={'xx': 'l'}))
        return  # df1, df2, df3

    def findxx(self, low, high, filterlist=('',), kl='wk', cc='bk', align={}):
        posfield = 'wkpos' if kl == 'wk' else 'lkpos'
        if cc == 'bk':
            # print('2016pc1---')
            df1 = self.get_df_from_pos(self.td16p1, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            # print('2016pc2---')
            df2 = self.get_df_from_pos(self.td16p2, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            # print('2017---')
            df3 = self.get_df_from_pos(self.td17bk, lowpos=low, highpos=high, posfield=posfield,
                                       filterlist=filterlist, kl=kl)
            df3.loc[:, 'xxh'] = df3.xx.apply(lambda x: str(x)[0:4])
            if kl == 'lk':
                df1 = df1.rename(columns={'lkjh': 'lkjh16', 'lkpos': 'lkpos16'})
                df2 = df2.rename(columns={'lkjh': 'lkjh16p2', 'lkpos': 'lkpos16p2'})
                df3 = df3.rename(columns={'lkjh': 'lkjh17', 'lkpos': 'lkpos17'})
                outfields = ['xx', 'lkjh17', 'lkjh16', 'lkjh16p2', 'lkpos17', 'lkpos16', 'lkpos16p2']
            else:
                df1 = df1.rename(columns={'wkjh': 'wkjh16', 'wkpos': 'wkpos16'})
                df2 = df2.rename(columns={'wkjh': 'wkjh16p2', 'wkpos': 'wkpos16p2'})
                df3 = df3.rename(columns={'wkjh': 'wkjh17', 'wkpos': 'wkpos17'})
                outfields = ['xx', 'wkjh17', 'wkjh16', 'wkjh16p2', 'wkpos17', 'wkpos16', 'wkpos16p2']
            dfmerge = pd.merge(df3, df1, on='xx', how='outer')
            dfmerge = pd.merge(dfmerge, df2, on='xx', how='outer')[outfields]
            dfmerge = dfmerge.fillna('0')
            if kl == 'lk':
                dfmerge = dfmerge.astype(dtype={'lkpos17': int, 'lkpos16': int, 'lkpos16p2': int,
                                                'lkjh17': int, 'lkjh16': int, 'lkjh16p2': int
                                                }, errors='ignore')
            else:
                dfmerge = dfmerge.astype(dtype={'wkpos17': int, 'wkpos16': int, 'wkpos16p2': int,
                                                'wkjh17': int, 'wkjh16': int, 'wkjh16p2': int
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
        print(make_page(dfmerge, title='16-17：'+ cc, align=align))
        return  dfmerge

    def findzy(self, lowpos, highpos, zyfilterlist, xxfilterlist, kl='lk'):
        if self.dflq is None:
            return pd.DataFrame()
        kldm = 1 if kl == 'wk' else 5
        filterfunzy = closed_filter(zyfilterlist)
        filterfunxx = closed_filter(xxfilterlist)
        df = self.dflq[self.dflq.ZYMC.apply(filterfunzy) & (self.dflq.YXMC.apply(filterfunxx)) &
                       (self.dflq.WC >= lowpos) & (self.dflq.WC <= highpos) & (self.dflq.KLDM == kldm)].\
            groupby(['YXDH', 'ZYDH'])[['WC', 'YXMC', 'ZYMC']].max()
        if len(df) == 0:
            print(' no result!')
            return
        print(make_page(df.sort_values('WC'),
                        title=''.join(zyfilterlist) + '(' + kl + ')',
                        align={'YXMC': 'l', 'ZYMC': 'l'}))
        return  # df

    def get_df_from_pos(self, df, lowpos, highpos, posfield, filterlist, kl):
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


def make_table(df, title='', align={}):
    x = Ptt()
    j = 0
    for f in df.columns:
        x.add_column(f, [x for x in df[f]])
        if (f in align):
            if (align[f] in ['l', 'c', 'r']):
                x.align[f] = align[f]
            elif df[f]._is_numeric_mixed_type:
                x.align[f] = 'r'
            elif df[f]._is_mixed_type:
                x.align[f] = 'l'
            else:
                x.align[f] = 'c'
        j = j + 1
    rs = x.get_string()
    return title.center(rs.index('\n')) + '\n' + rs


def make_page(df, title='', pagelines=30, align={}):
    gridnum = len(df.columns)
    result = ''
    ptext = make_table(df=df, title=title, align=align)
    plist = ptext.split('\n')
    # print(plist)
    plen = len(plist)
    hline = 0
    textline = 0
    head = ''
    gapline = None
    pagewid = 0
    pageno = 0
    for i in range(plen):
        if hline < 2:
            # set subtitle in center
            if ('+' not in plist[i]) & (plist[i].count('|') == gridnum + 1):
                sp = plist[i].split('|')
                newsp = []
                for x in sp:
                    if len(x.strip()) < len(x):
                        left_space = int((len(x) - len(x.strip()))/2)
                        newsp.append(' '*left_space + x.strip() + ' '*(len(x) - left_space-len(x.strip())))
                head += '|' + '|'.join(newsp) + '|\n'
            else:
                head += plist[i] + '\n'
        else:
            # not save first head in result
            if i < plen -1:
                result += plist[i] + '\n'
        # find gapline and the end of head
        if plist[i].count('+') == gridnum + 1:
            hline = hline + 1
            if gapline is None:
                pagewid = len(plist[i])
                gapline = plist[i] + '\n'
            continue
        # add first head+gapline in result
        if (len(result) == 0) & (gapline is not None):
            result = head + gapline
        # start count content row number(textline)
        if hline == 2:
            textline += 1
        # seperate pages
        if (textline == pagelines) | (i == plen-2):
            pageno += 1
            pagenostr = ('--'+str(pageno)+'--').center(pagewid) + '\n\n'
            result += gapline + pagenostr + (head if i < plen-2 else '')
            textline = 0
    return result
