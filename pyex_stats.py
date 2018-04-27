# -*- utf-8 -*-
# version 2017-09-16

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import pyex_lib as pl
import pyex_seg as ps

# ywmean = df.yw.mean(), wlmean = df.wl.mean()
# m = math.sqrt(sum([(x - ywmean)**2 for x in df.yw]))*math.sqrt(sum([(x - wlmean)**2 for x in df.wl]))
# c = sum([(x - ywmean)*(y - wlmean) for x, y in zip(df.yw, df.wl)])
# pearsonr = c/m
def relation(x, y):
    plt.scatter(x, y)
    return stats.pearsonr(x, y)[0]


def float_str(x, d1, d2):
    return f'%{d1}.{d2}f' % x


def int_str(x,d):
    return f'%{d}d' % x


def exp_r(noise=10):
    tf = pl.exp_scoredf_normal(mean=60, std=10, size=1000)
    tf['sf2'] = tf.sf.apply(lambda v: v + np.random.rand()*noise)
    rs = relation(tf.sf, tf.sf2)
    maxdiff = max(abs(tf.sf - tf.sf2))
    #plt.figure()
    plt.scatter(tf.sf, tf.sf2, label='relation')
    plt.title('noise={n}   PearsonR={r}   MaxDiff={d}'.
              format(n=noise, r=float_str(rs, 2, 4), d=float_str(maxdiff, 2, 4)))


class ScoreData():
    """
    read gk data from csv
    include kl, ysw, wl, hx, sw
    """

    def __init__(self):
        self.fs_file = 'd:/work/newgk/shanghai1711/cj17b.csv'
        self.rawdf = None
        self.dfwlgsw = None
        self.dfswgwl = None
        self.dfyswgwl = None

    def read_rawdf(self):
        # self.df = pd.read_csv('d:/work/newgk/shanghai1711/cj17b.csv', sep='\t', index_col=0)
        self.rawdf = pd.read_csv(self.fs_file, sep='\t', index_col=0)
        # self.rawdf = self.rawdf[self.rawdf.ysw > 0]
        # self.dfswgwl = pd.DataFrame({'wl': [self.rawdf[self.rawdf.sw == x]['wl'].mean() for x in range(91)], 'sw': [x for x in range(91)]})
        # self.dfwlgsw = pd.DataFrame({'sw': [self.rawdf[self.rawdf.wl == x]['sw'].mean() for x in range(111)], 'wl': [x for x in range(111)]})
        # self.dfyswgwl = pd.DataFrame({'wl': [self.rawdf[self.rawdf.ysw == x]['wl'].mean() for x in range(450)], 'ysw': [x for x in range(450)]})
        # self.dfwlgysw = pd.DataFrame({'ysw': [self.rawdf[self.rawdf.wl == x]['ysw'].mean() for x in range(111)], 'wl': [x for x in range(111)]})
        return


def df_format(dfsource, intlen=2, declen=4, strlen=8):
    df = dfsource[[dfsource.columns[0]]]
    fdinfo = dfsource.dtypes
    for fs in fdinfo.index:
        if fdinfo[fs] in [np.float, np.float16, np.float32, np.float64]:
            df[fs+'_str'] = dfsource[fs].apply(lambda x: float_str(x, intlen, declen))
        elif fdinfo[fs] in [np.int, np.int8, np.int16, np.int32, np.int64]:
            df[fs+'_str'] = dfsource[fs].apply(lambda x: int_str(x, 6))
        elif fdinfo[fs] in [str]:
            df[fs+'_fmt'] = dfsource[fs].apply(lambda x: x.rjust(strlen))
    df.sort_index(axis=1)
    return df


def ref_stm(df, fkey, f1, f2, adj_rate_points=(0.35, 0.75)):
    segmodel = ps.SegTable()
    segmodel.set_data(df, [f1, f2])
    segmodel.set_parameters(segmax=max(df[f1]))
    segmodel.run()
    segf1 = segmodel.segdf
    segmodel.set_parameters(segmax=max(df[f2]))
    segf2 = segmodel.segdf
    f1points = []
    for p in adj_rate_points:
        f2count = segf2.loc[segf2[f2+'_percent'] >= p, f2+'_cumsum'].head(1)['seg']


def cross_seg(df, keyf,
              vf, vfseglist=(50, 60, 70, 80, 90, 100)):
    segmodel = ps.SegTable()
    segmodel.set_data(df, keyf)
    segmodel.set_parameters(segmax=max(df[keyf]))
    segmodel.run()
    dfseg = segmodel.segdf
    dfcount = dfseg[keyf+'_cumsum'].tail(1).values[0]
    vfseg = {x:[] for x in vfseglist}
    vfper = {x:[] for x in vfseglist}
    seglen = dfseg['seg'].count()
    for sv, step in zip(dfseg['seg'], range(seglen)):
        if (step % 20 == 0) | (step == seglen-1):
            print('='* int((step+1)/seglen * 30) + '>>' + f'{float_str((step+1)/seglen, 1, 2)}')
        segv = []
        for vfv in vfseglist:
            segcount = df.loc[(df[keyf] >= sv) & (df[vf] >= vfv), vf].count()
            vfseg[vfv].append(segcount)
            vfper[vfv].append(segcount/dfcount)
    for vs in vfseglist:
        dfseg[vf + str(vs) + '_cumsum'] = vfseg[vs]
        dfseg[vf + str(vs) + '_percent'] = vfper[vs]
    return dfseg
