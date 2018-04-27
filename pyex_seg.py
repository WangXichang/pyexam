# -*- utf-8 -*-
# version 2017-09-16

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
# import matplotlib as mp
# from texttable import Texttable


# test SegTable
def test_segtable():
    """
    a example for test SegTable
    ---------------------------------------------------------------------------
    expdf = pd.DataFrame({'sf': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9]})
    seg = SegTable()
    seg.set_data(expdf, expdf.columns.values)
    seg.set_parameters(segstep=3, segmax=8, segmin=3, segclip=False)
    seg.run()
    print(seg.segdf)
       seg  sf_count  sf_cumsum  sf_percent  sf_count3
    5    8         2          2    0.133333          2
    4    7         1          3    0.200000         -1
    3    6         1          4    0.266667         -1
    2    5         1          5    0.333333          3
    1    4         4          9    0.600000         -1
    0    3         6         15    1.000000         10
    ------------------------------------------------------------------------------
    """
    expdf = pd.DataFrame({'sf': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9]})
    seg = SegTable()
    seg.set_data(expdf, ['sf'])
    seg.set_parameters(segstep=3, segmax=8, segmin=3, segalldata=True, dispmode=True)
    seg.run()
    seg.plot()
    seg.show_parameters()
    print(seg.segdf)
    # change parameters to run to get new result
    seg.segalldata = False
    seg.segmax = 7
    seg.segstep = 2
    seg.run()
    seg.show_parameters()
    print(seg.segdf)
    return seg


# 计算pandas.DataFrame中分数字段的分段人数表
# segment table for score dataframe
# version 0917-2017
class SegTable(object):
    """
    设置数据，数据表（类型为pandas.DataFrame）,同时需要设置需要计算分数分段人数的字段（list类型）
    :data
        rawdf: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        segfields: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的分数字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   分数字段的类型应为可计算类型，如int,float.
    设置参数
    :parameters
        segmax: int,  maxvalue for segment, default=150输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。输出分段表中分数段的最小值
        segstep: int, levels for segment value, default=1
                 分段间隔，用于生成n-分段表（五分一段的分段表）
        segsort: str, 'ascending' or 'descending', default='descending'(sort seg descending)
                 输出结果中分段值得排序方式，descending:从大到小， ascending：从小到大
                 排序模式的设置影响累计数和百分比的意义。
        segalldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        dispmode: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息
    运行结果
    :result
        segdf: dataframe with field 'seg, segfield_count, segfield_cumsum, segfield_percent'
    应用举例
    example:
        import pyex_seg
        seg = pyex_seg.SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, 'sf')
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='descending', segalldata=True, dispmode=True)
        seg.run()
        seg.plot()
        resultdf = seg.segdf    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent
    备注
    Note:
        1)根据segclip确定是否在设定的区间范围内计算分数值，segclip=True时抛弃不再范围内的分数项
        segclip=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
        segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
        segclip is used to include or exclude count() outside to segmin or segmax repectively
        2)分数字段的类型为整数或浮点数（实数）
        score fields type is int or float
        3)可以通过属性方式单独设置数据(rawdata),字段列表（scorefields),各项参数（segmax, segmin, segsort,segalldata,
        segmode), 如，seg.scorefields = ['score_1', 'score_2']; seg.segmax = 120， 便于在计算期间调整模型。
        by property mode, rawdata,scorefields,parameters can be setted individually
    """

    def __init__(self):
        # raw data
        self.__rawDf = None
        self.__segFields = []
        # parameter for model
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'descending'
        self.__segAlldata = True
        self.__disp = False
        # result data
        self.__segDf = None
        # run status
        self.__runsuccess = False

    @property
    def segdf(self):
        return self.__segDf

    @property
    def rawdf(self):
        return self.__rawDf

    @rawdf.setter
    def rawdf(self, df):
        self.__rawDf = df

    @property
    def segfields(self):
        return self.__segFields

    @segfields.setter
    def segfields(self, sfs):
        self.__segFields = sfs

    @property
    def segmax(self):
        return self.__segMax

    @segmax.setter
    def segmax(self, segvalue):
        self.__segMax = segvalue

    @property
    def segmin(self):
        return self.__segMin

    @segmin.setter
    def segmin(self, segvalue):
        self.__segMin = segvalue

    @property
    def segsort(self):
        return self.__segSort

    @segsort.setter
    def segsort(self, sortstr):
        self.__segSort = sortstr

    @property
    def segalldata(self):
        return self.__segAlldata

    @segalldata.setter
    def segalldata(self, datamode):
        self.__segAlldata = datamode

    @property
    def dispmode(self):
        return self.__disp

    @dispmode.setter
    def dispmode(self, dispmode):
        self.__disp = dispmode

    def set_data(self, df, segfields=None):
        self.rawdf = df
        if type(segfields) == str:
            segfields = [segfields]
        if (type(segfields) != list) & (type(df) == pd.DataFrame):
            self.segfields = df.columns.values
        else:
            self.segfields = segfields

    def set_parameters(self, segmax=100, segmin=0, segstep=1, segsort='descending',
                       segalldata=False, dispmode=True):
        self.__segMax = segmax
        self.__segMin = segmin
        self.__segStep = segstep
        self.__segSort = segsort
        self.__segAlldata = segalldata
        self.__disp = dispmode

    def show_parameters(self):
        print('seg max value:{}'.format(self.__segMax))
        print('seg min value:{}'.format(self.__segMin))
        print('seg step value:{}'.format(self.__segStep))
        print('seg sort mode:{}'.format(self.__segSort))
        print('seg crop mode:{}'.format(self.__segAlldata))
        print('seg disp mode:{}'.format(self.__disp))

    def check(self):
        if type(self.__rawDf) == pd.Series:
            self.__rawDf = pd.DataFrame(self.__rawDf)
        if type(self.__rawDf) != pd.DataFrame:
            print('data set is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('segmax value is not greater than segmin!')
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('segstep is too small or big!')
            return False
        if type(self.segfields) != list:
            if type(self.segfields) == str:
                self.segfields = [self.segfields]
            else:
                print('segfields error:', type(self.segfields))
                return False
            for f in self.segfields:
                if f not in self.rawdf.columns.values:
                    print('field in segfields is not in rawdf:', f)
                    return False
        if type(self.__segAlldata) != bool:
            print('segclip is not bool type!')
            return False
        return True

    def run(self):
        sttime = time.clock()
        if not self.check():
            return
        # create output dataframe with segstep = 1
        seglist = [x for x in range(self.__segMin, self.__segMax + 1)]
        self.__segDf = pd.DataFrame({'seg': seglist})
        for f in self.segfields:
            # calculate preliminary group count
            r = self.rawdf.groupby(f)[f].count()
            if self.__disp:
                print('finished groupby ' + f, ' use time:{0}'.format(time.clock() - sttime))
            # count seg_count in [segmin, segmax]
            self.__segDf[f + '_count'] = self.__segDf['seg'].\
                apply(lambda x: np.int64(r[x]) if x in r.index else 0)
            if self.__disp:
                print('finished count ' + f, ' use time:{}'.format(time.clock() - sttime))
            # add outside scope number to segmin, segmax
            if self.__segAlldata:
                self.__segDf.loc[self.__segDf.seg == self.__segMin, f+'_count'] = r[r.index <= self.__segMin].sum()
                self.__segDf.loc[self.__segDf.seg == self.__segMax, f+'_count'] = r[r.index >= self.__segMax].sum()
            # set order for seg fields
            if self.__segSort != 'ascending':
                self.__segDf = self.__segDf.sort_values(by='seg', ascending=False)
            # calculate cumsum field
            self.__segDf[f + '_cumsum'] = self.__segDf[f + '_count'].cumsum()
            # calculate percent field
            if self.__disp:
                print('finished cumsum ' + f, ' use time:{0}'.format(time.clock() - sttime))
            maxsum = max(max(self.segdf[f + '_cumsum']), 1)     # avoid divided by 0 in percent computing
            self.__segDf[f + '_percent'] = self.__segDf[f + '_cumsum'].apply(lambda x: x / maxsum)
            if self.__disp:
                print('finished percent ' + f, ' use time:{}'.format(time.clock() - sttime))
            # processing seg step calculating: skip step at seg field, set -1 for segs not in step
            if self.__segStep > 1:
                segcountname = f+'_count{0}'.format(self.__segStep)
                self.__segDf[segcountname] = np.int64(-1)
                c = 0
                curpoint, curstep = ((self.__segMin, self.__segStep)
                                     if self.__segSort == 'ascending' else
                                     (self.__segMax, -self.__segStep))
                for index, row in self.__segDf.iterrows():
                    c += row[f+'_count']
                    if np.int64(row['seg']) in [curpoint, self.__segMax, self.__segMin]:
                        # row[segcountname] = c
                        self.__segDf.loc[index, segcountname] = np.int64(c)
                        c = 0
                        curpoint += curstep
        if self.__disp:
            print('total consumed time:{}'.format(time.clock()-sttime))
        self.__runsuccess = True
        return

    def plot(self):
        if not self.__runsuccess:
            if self.__disp:
                print('result is not created, please run!')
            return
        legendlist = []
        step = 0
        for sf in self.segfields:
            step += 1
            legendlist.append(sf)
            plt.figure('seg table figure({})'.format(self.__segSort))
            plt.subplot(221)
            plt.hist(self.rawdf[sf], 20)
            plt.title('raw data histogram')
            if step == len(self.segfields):
                plt.legend(legendlist)
            plt.subplot(222)
            plt.plot(self.segdf.seg, self.segdf[sf+'_count'])
            if step == len(self.segfields):
                plt.legend(legendlist)
            plt.title('seg -- count')
            plt.xlim([self.__segMin, self.__segMax])
            plt.subplot(223)
            plt.plot(self.segdf.seg, self.segdf[sf + '_cumsum'])
            plt.title('seg -- cumsum')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.segfields):
                plt.legend(legendlist)
            plt.subplot(224)
            plt.plot(self.segdf.seg, self.segdf[sf + '_percent'])
            plt.title('seg -- percent')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.segfields):
                plt.legend(legendlist)
            plt.show()
# SegTable class end


def cross_seg(df,    # source dataframe
              keyf,  # key field to calculate segment
              vf,  # cross field, calculate count for >=keyf_seg & >=vf_seg
              vfseglist=(40, 50, 60, 70, 80, 90, 100)  # segment for cross field
              ):
    display_step = 20
    segmodel = SegTable()
    segmodel.set_data(df, keyf)
    segmodel.set_parameters(segmax=max(df[keyf]))
    segmodel.run()
    dfseg = segmodel.segdf
    dfcount = dfseg[keyf+'_cumsum'].tail(1).values[0]
    vfseg = {x: [] for x in vfseglist}
    vfper = {x: [] for x in vfseglist}
    seglen = dfseg['seg'].count()
    for sv, step in zip(dfseg['seg'], range(seglen)):
        if (step % display_step == 0) | (step == seglen-1):
            print('=' * int((step+1)/seglen * 30) + '>>' + f'{float_str((step+1)/seglen, 1, 2)}')
        for vfv in vfseglist:
            segcount = df.loc[(df[keyf] >= sv) & (df[vf] >= vfv), vf].count()
            vfseg[vfv].append(segcount)
            vfper[vfv].append(segcount/dfcount)
    for vs in vfseglist:
        dfseg[vf + str(vs) + '_cumsum'] = vfseg[vs]
        dfseg[vf + str(vs) + '_percent'] = vfper[vs]
    return dfseg


def float_str(x, d1, d2):
    d1 = d1 + d2 + 1
    return f'%{d1}.{d2}f' % x


def int_str(x, d):
    return f'%{d}d' % x
