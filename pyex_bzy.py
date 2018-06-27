# -*- utf -*-

import pandas as pd
import pyex_tab as pt


class Zy:
    def __init__(self):
        self.td16p1 = pd.read_csv('f:/studies/lqdata/td2016pc1_sc.csv', sep=',', dtype={'xx': str}, verbose=True)
        self.td16p2 = pd.read_csv('f:/studies/lqdata/td2016pc2_sc.csv', sep=',', dtype={'xx': str}, verbose=True)
        self.td17bk = pd.read_csv('f:/studies/lqdata/td2017bk_sc.csv', sep=',', dtype={'xx': str}, verbose=True)

    def lk(self, low, high, filter=('')):
        filterfun = self.filtfun(filter)
        print('2016pc1---')
        print(self.td16p1[['xx', 'lkpos']][(self.td16p1.lkpos <= high) & (self.td16p1.lkpos >= low) &
              # self.td16p1.xx.apply(lambda x: filter in str(x))
              self.td16p1.xx.apply(lambda x: filterfun(x))
              ])
        print('2016pc2---')
        print(self.td16p2[['xx', 'lkpos']][(self.td16p2.lkpos <= high) & (self.td16p2.lkpos >= low) &
              #self.td16p2.xx.apply(lambda x: filter in str(x))
              self.td16p2.xx.apply(lambda x: filterfun(str(x)))
              ])
        print('2017---')
        print(self.td17bk[['xx', 'lkpos']][(self.td17bk.lkpos <= high) & (self.td17bk.lkpos >= low) &
              # self.td17bk.xx.apply(lambda x: filter in str(x))
              self.td17bk.xx.apply(lambda x: filterfun(str(x)))
              ])

    def find_xx(self, low, high, filterstr='', kl='wk'):
        posfield = 'wkpos' if kl == 'wk' else 'lkpos'
        print('2016pc1---')
        df1 = self.get_df_from_pos(self.td16p1, lowpos=low, highpos=high, posfield=posfield, filterstr=filterstr)
        print(pt.df_to_table(df1))
        # print(df1)
        print('2016pc2---')
        df2 = self.get_df_from_pos(self.td16p2, lowpos=low, highpos=high, posfield=posfield, filterstr=filterstr)
        print(df2)
        print('2017---')
        df3 = self.get_df_from_pos(self.td17bk, lowpos=low, highpos=high, posfield=posfield, filterstr=filterstr)
        print(df3)
        return df1, df2, df3

    def get_df_from_pos(self, df,lowpos, highpos, posfield, filterstr):
        return df[['xx', posfield]][(df[posfield] <= highpos) & (df[posfield] >= lowpos) &
                   df.xx.apply(lambda x: filterstr in str(x))].sort_values(by=posfield)

    def somexx(self, filter='医学', kl='wk'):
        print('2016p1---')
        print(self.td16p1[self.td16p1.xx.apply(lambda x: filter in str(x))][['xx', 'wkpos', 'lkpos']].
              sort_values(by=('lkpos' if kl == 'lk' else 'wkpos')))
        print('2016p2---')
        print(self.td16p2[self.td16p2.xx.apply(lambda x: filter in str(x))][['xx', 'wkpos', 'lkpos']].
              sort_values(by='lkpos' if kl == 'lk' else 'wkpos'))
        print('2017bk---')
        print(self.td17bk[self.td17bk.xx.apply(lambda x: filter in str(x))][['xx', 'wkpos', 'lkpos']].
              sort_values(by='lkpos' if kl == 'lk' else 'wkpos'))

    def filtfun(self, filterstr):
        filterstr = filterstr
        if (not isinstance(filterstr, list)) & (not isinstance(filterstr, tuple)):
            print('filter is not list')
            return False
        def filter(x):
            result = False
            for s in filterstr:
                if s in x:
                    result = True
                    break
            return result
        return filter
