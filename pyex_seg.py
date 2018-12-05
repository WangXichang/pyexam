# -*- utf-8 -*-
# from 2017-09-16
# version 1.0.2     at 2018-6-24

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
# import copy


# guid to use pyex_seg
def doc():
    print(SegTable.__doc__)
    print(test.__doc__)


# test SegTable and show some example
def test():
    """
    a example for test SegTable
    ---------------------------------------------------------------------------
    expdf = pd.DataFrame({'sf': [1, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, ]})
    seg = SegTable()
    seg.set_data(expdf, expdf.columns.values)
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=3, usealldata=True, display=True)
    seg.run()
    print(seg.output_data)
         seg      sf_count  sf_sum  sf_percent  sf_count3
    5    8         2          2    0.133333          2
    4    7         1          3    0.200000         -1
    3    6         1          4    0.266667         -1
    2    5         1          5    0.333333          3
    1    4         4          9    0.600000         -1
    0    3         8         17    1.000000         12  # including 0, 1 which below 3
    ------------------------------------------------------------------------------
    # avoid to count records below segmin if segalldata = False
    seg.set_parameters(segstep=3, segstart=8, segmax=7, segmin=0, usealldata=False, display=True)
    seg.run()
    print(seg.output_data)
    ----------------------------
       seg  sf_count  sf_sum  sf_percent  sf_count3
    0    7         1       1    0.066667          1  # excluding 8, 8
    1    6         1       2    0.133333         -1
    2    5         1       3    0.200000          2
    3    4         4       7    0.466667         -1
    4    3         6      13    0.866667         -1
    5    2         0      13    0.866667         10
    6    1         1      14    0.933333         -1
    7    0         1      15    1.000000          2
    """

    expdf = pd.DataFrame({'sf': [1, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8]})
    print('example dataframe')
    print('='*80)
    print(expdf)

    seg = SegTable()
    seg.set_data(expdf, ['sf'])
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=0, usealldata=True, display=False)
    seg.run()
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)

    # special step, start, no all data
    seg.set_parameters(segstep=3, segstart=8, segmax=7, segmin=0, usealldata=False, display=False)
    seg.run()
    print('\n')
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)

    # change parameters to run to get new result
    seg.segalldata = True
    seg.segstart = 7
    seg.segmax = 10
    seg.segmin = 1
    seg.segstep = 2
    seg.display = False
    seg.run()
    print('\n')
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)

    # change parameters to run to get new result
    seg.useseglist = True
    seg.seglist = [7, 4, 3, 1]
    seg.segstart = 5
    seg.segmax = 7
    seg.segsort = 'd'
    seg.display = False
    seg.run()
    print('\n')
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)
    print('\n--- get result use: (df.sf_list_sum > 0) | (df.seg.isin(range(segmin, segmax+1)) ---')
    print(seg.output_data[seg.output_data.sf_list > 0][['seg', 'sf_list', 'sf_list_sum']])
    print(seg.output_data[seg.output_data.sf_count2 > 0][['seg', 'sf_count2', 'sf_sum']])

    return seg
    # end test


def run(df=None,
        field_list=None,
        segmax=150,
        segmin=0,
        segstart=150,
        seglist=None,
        useseglist=False,
        segalldata=True,
        segstep=1,
        segsort='d',
        display=True):
    if not isinstance(df, pd.DataFrame):
        print('df is not DataFrame!')
        return
    if type(field_list) != list:
        print('field_list is not a list!')
        return
    for fs in field_list:
        if fs not in df.columns:
            print('{0} is not in df.columns!'.format(fs))
            return
    sm = SegTable()
    sm.set_data(input_data=df, field_list=field_list)
    sm.set_parameters(segmax=segmax,
                      segmin=segmin,
                      segstep=segstep,
                      segstart=segstart,
                      segsort=segsort,
                      seglist=seglist,
                      useseglist=useseglist,
                      usealldata=segalldata,
                      display=display
                      )
    sm.run()
    return sm


class SegTable(object):
    """
    * 计算pandas.DataFrame中分数字段的分段人数表
    * segment table for score dataframe
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    * from 09-17-2017

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(input_data:DataFrame, field_list:list)
        input_data: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        field_list: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离，分段开始值，分数顺序，指定分段值列表， 使用指定分段列表，使用所有数据， 关闭计算过程显示信息
    set_parameters（segmax, segmin, segstep, segstart, segsort, seglist, useseglist, usealldata, display）
        segmax: int, maxvalue for segment, default=150
                输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。
                输出分段表中分数段的最小值
        segstep: int, levels for segment value, default=1
                分段间隔，用于生成n-分段表（五分一段的分段表）
        segstart:int, start seg score to count
                进行分段计算的起始值
        segsort: str, 'a' for ascending order or 'd' for descending order, default='d' (seg order on descending)
                输出结果中分段值得排序方式，d: 从大到小， a：从小到大
                排序模式的设置影响累计数和百分比的意义。
        seglist: list, used to create set value
                 使用给定的列表产生分段表，列表中为分段点值
        useseglist: bool, use or not use seglist to create seg value
                 是否使用给定列表产生分段值
        usealldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        display: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息
    output_data: 输出分段数据
            seg: seg value
        [field]: field name in field_list
        [field]_count: number at the seg
        [field]_sum: cumsum number at the seg
        [field]_percent: percentage at the seg
        [field]_count[step]: count field for step != 1
        [field]_list: count field for assigned seglist when use seglist
    运行，产生输出数据, calculate and create output data
    run()

    应用举例
    example:
        import pyex_seg as sg
        seg = sg.SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='d', usealldata=True, display=True)
        seg.run()
        seg.plot()
        print(seg.output_data.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          field_list type is digit, for example: int or float

        3)可以单独设置数据(input_data),字段列表（field_list),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.field_list = ['score_1', 'score_2'];
              seg.segmax = 120
          重新设置后需要运行才能更新输出数据ouput_data, 即调用run()
          便于在计算期间调整模型。
          by usting property mode, rawdata, scorefields, parameters can be setted individually
        4) 当设置大于1分的分段分值X时， 会在结果DataFrame中生成一个字段[segfiled]_countX，改字段中不需要计算的分段
          值设为-1。
          when segstep > 1, will create field [segfield]_countX, X=str(segstep), no used value set to -1 in this field
    """

    def __init__(self):
        # raw data
        self.__input_dataframe = None
        self.__segFields = []
        # parameter for model
        self.__segList = []
        self.__useseglist = False
        self.__segStart = 100
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'd'
        self.__usealldata = True
        self.__display = True
        self.__percent_decimal = 8
        # result data
        self.__output_dataframe = None
        # run status
        self.__run_completed = False

    @property
    def output_data(self):
        return self.__output_dataframe

    @property
    def input_data(self):
        return self.__input_dataframe

    @input_data.setter
    def input_data(self, df):
        self.__input_dataframe = df

    @property
    def field_list(self):
        return self.__segFields

    @field_list.setter
    def field_list(self, field_list):
        self.__segFields = field_list

    @property
    def seglist(self):
        return self.__segList

    @seglist.setter
    def seglist(self, seglist):
        self.__segList = seglist

    @property
    def useseglist(self):
        return self.__useseglist

    @useseglist.setter
    def useseglist(self, useseglist):
        self.__useseglist = useseglist

    @property
    def segstart(self):
        return self.__segStart

    @segstart.setter
    def segstart(self, segstart):
        self.__segStart = segstart

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
    def segsort(self, sort_mode):
        self.__segSort = sort_mode

    @property
    def segstep(self):
        return self.__segStep

    @segstep.setter
    def segstep(self, segstep):
        self.__segStep = segstep

    @property
    def segalldata(self):
        return self.__usealldata

    @segalldata.setter
    def segalldata(self, datamode):
        self.__usealldata = datamode

    @property
    def display(self):
        return self.__display

    @display.setter
    def display(self, display):
        self.__display = display

    def set_data(self, input_data, field_list=None):
        self.input_data = input_data
        if type(field_list) == str:
            field_list = [field_list]
        if (not isinstance(field_list, list)) & isinstance(input_data, pd.DataFrame):
            self.field_list = input_data.columns.values
        else:
            self.field_list = field_list
        self.__check()

    def set_parameters(
            self,
            segmax=None,
            segmin=None,
            segstart=None,
            segstep=None,
            seglist=None,
            segsort=None,
            useseglist=None,
            usealldata=None,
            display=None):
        set_str = ''
        if isinstance(segmax, int):
            self.__segMax = segmax
            set_str += 'set segmax to {}'.format(segmax) + '\n'
        if isinstance(segmin, int):
            self.__segMin = segmin
            set_str += 'set segmin to {}'.format(segmin) + '\n'
        if isinstance(segstep, int):
            self.__segStep = segstep
            set_str += 'set segstep to {}'.format(segstep) + '\n'
        if isinstance(segsort, str):
            if segsort.lower() in ['d', 'a', 'D', 'A']:
                set_str += 'set segsort to {}'.format(segsort) + '\n'
                self.__segSort = segsort
        if isinstance(usealldata, bool):
            set_str += 'set segalldata to {}'.format(usealldata) + '\n'
            self.__usealldata = usealldata
        if isinstance(display, bool):
            set_str += 'set display to {}'.format(display) + '\n'
            self.__display = display
        if isinstance(segstart, int):
            set_str += 'set segstart to {}'.format(segstart) + '\n'
            self.__segStart = segstart
        if isinstance(seglist, list):
            set_str += 'set seglist to {}'.format(seglist) + '\n'
            self.__segList = seglist
        if isinstance(useseglist, bool):
            set_str += 'set seglistuse to {}'.format(useseglist) + '\n'
            self.__useseglist = useseglist
        if display:
            print(set_str)
        self.__check()
        if display:
            self.show_parameters()

    def show_parameters(self):
        print('------ seg parameters ------')
        print('    use seglist:{0}'.format(self.__useseglist, self.__segList))
        print('        seglist:{1}'.format(self.__useseglist, self.__segList))
        print('       maxvalue:{}'.format(self.__segMax))
        print('       minvalue:{}'.format(self.__segMin))
        print('       segstart:{}'.format(self.__segStart))
        print('        segstep:{}'.format(self.__segStep))
        print('        segsort:{}'.format('d (descending)' if self.__segSort in ['d', 'D'] else 'a (ascending)'))
        print('     usealldata:{}'.format(self.__usealldata))
        print('        display:{}'.format(self.__display))
        print('-' * 28)

    def helpdoc(self):
        print(self.__doc__)

    def __check(self):
        if isinstance(self.__input_dataframe, pd.Series):
            self.__input_dataframe = pd.DataFrame(self.__input_dataframe)
        if not isinstance(self.__input_dataframe, pd.DataFrame):
            print('error: raw score data is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('error: segmax({}) is not greater than segmin({})!'.format(self.__segMax, self.__segMin))
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('error: segstep({}) is too small or big!'.format(self.__segStep))
            return False
        if not isinstance(self.field_list, list):
            if isinstance(self.field_list, str):
                self.field_list = [self.field_list]
            else:
                print('error: segfields type({}) error.'.format(type(self.field_list)))
                return False

        for f in self.field_list:
            if f not in self.input_data.columns:
                print("error: field('{}') is not in input_data fields({})".
                      format(f, self.input_data.columns.values))
                return False
        if not isinstance(self.__usealldata, bool):
            print('error: segalldata({}) is not bool type!'.format(self.__usealldata))
            return False
        return True

    def run(self):
        sttime = time.clock()
        if not self.__check():
            return
        # create output dataframe with segstep = 1
        if self.__display:
            print('seg calculation start ...')
        seglist = [x for x in range(self.__segMin, self.__segMax + 1)]
        if self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__output_dataframe = pd.DataFrame({'seg': seglist})
        outdf = self.__output_dataframe
        for f in self.field_list:
            # calculate preliminary group count
            tempdf = self.input_data
            tempdf.loc[:, f] = tempdf[f].apply(round45i)

            # count seg_count in [segmin, segmax]
            r = tempdf.groupby(f)[f].count()
            fcount_list = [np.int64(r[x]) if x in r.index else 0 for x in seglist]

            outdf.loc[:, f+'_count'] = fcount_list
            if self.__display:
                print('finished count(' + f, ') use time:{}'.format(time.clock() - sttime))

            # add outside scope number to segmin, segmax
            if self.__usealldata:
                outdf.loc[outdf.seg == self.__segMin, f + '_count'] = \
                    r[r.index <= self.__segMin].sum()
                outdf.loc[outdf.seg == self.__segMax, f + '_count'] = \
                    r[r.index >= self.__segMax].sum()

            # calculate cumsum field
            outdf[f + '_sum'] = outdf[f + '_count'].cumsum()
            if self.__useseglist:
                outdf[f + '_list_sum'] = outdf[f + '_count'].cumsum()

            # calculate percent field
            maxsum = max(max(outdf[f + '_sum']), 1)     # avoid divided by 0 in percent computing
            outdf[f + '_percent'] = \
                outdf[f + '_sum'].apply(lambda x: round45i(x/maxsum, self.__percent_decimal))
            if self.__display:
                print('segments count finished[' + f, '], used time:{}'.format(time.clock() - sttime))

            # self.__output_dataframe = outdf.copy()
            # special seg step
            if self.__segStep > 1:
                self.__run_special_step(f)

            # use seglist
            if self.__useseglist:
                if len(self.__segList) > 0:
                    self.__run_seg_list(f)

        if self.__display:
            print('segments count total consumed time:{}'.format(time.clock()-sttime))
            print('=== end')
        self.__run_completed = True
        self.__output_dataframe = outdf
        return

    def __run_special_step(self, field: str):
        """
        processing count for step > 1
        :param field: for seg stepx
        :return: field_countx in output_data
        """
        f = field
        segcountname = f + '_count{0}'.format(self.__segStep)
        self.__output_dataframe[segcountname] = np.int64(-1)
        curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        curpoint = self.__segStart
        if self.__segSort.lower() == 'd':
            while curpoint+curstep > self.__segMax:
                curpoint += curstep
        else:
            while curpoint+curstep < self.__segMin:
                curpoint += curstep
        # curpoint = self.__segStart
        cum = 0
        for index, row in self.__output_dataframe.iterrows():
            cum += row[f + '_count']
            curseg = np.int64(row['seg'])
            if curseg in [self.__segMax, self.__segMin]:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if (self.__segStart <= self.__segMin) | (self.__segStart >= self.__segMax):
                    curpoint += curstep
                continue
            if curseg in [self.__segStart, curpoint]:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                curpoint += curstep

    def __run_seg_list(self, field):
        """
        use special step list to create seg
        calculating based on field_count
        :param field:
        :return:
        """
        f = field
        segcountname = f + '_list'
        self.__output_dataframe[segcountname] = np.int64(-1)
        segpoint = sorted(self.__segList) \
            if self.__segSort.lower() == 'a' \
            else sorted(self.__segList)[::-1]
        # curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        # curpoint = self.__segStart
        cum = 0
        pos = 0
        curpoint = segpoint[pos]
        rownum = len(self.__output_dataframe)
        cur_row = 0
        lastindex = 0
        maxpoint = max(self.__segList)
        minpoint = min(self.__segList)
        list_sum = 0
        self.__output_dataframe.loc[:, f+'_list_sum'] = 0
        for index, row in self.__output_dataframe.iterrows():
            curseg = np.int64(row['seg'])
            # cumsum
            if self.__usealldata | (minpoint <= curseg <= maxpoint):
                cum += row[f + '_count']
                list_sum += row[f+'_count']
                self.__output_dataframe.loc[index, f+'_list_sum'] = np.int64(list_sum)
            # set to seg count, only set seg in seglist
            if curseg == curpoint:
                self.__output_dataframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if pos < len(segpoint)-1:
                    pos += 1
                    curpoint = segpoint[pos]
                else:
                    lastindex = index
            elif cur_row == rownum:
                if self.__usealldata:
                    self.__output_dataframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1

    def plot(self):
        if not self.__run_completed:
            if self.__display:
                print('result is not created, please run!')
            return
        legendlist = []
        step = 0
        for sf in self.field_list:
            step += 1
            legendlist.append(sf)
            plt.figure('segtable figure({})'.
                       format('Descending' if self.__segSort in 'aA' else 'Ascending'))
            plt.subplot(221)
            plt.hist(self.input_data[sf], 20)
            plt.title('histogram')
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.subplot(222)
            plt.plot(self.output_data.seg, self.output_data[sf+'_count'])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.title('distribution')
            plt.xlim([self.__segMin, self.__segMax])
            plt.subplot(223)
            plt.plot(self.output_data.seg, self.output_data[sf + '_sum'])
            plt.title('cumsum')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.subplot(224)
            plt.plot(self.output_data.seg, self.output_data[sf + '_percent'])
            plt.title('percentage')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.show()

    # SegTable class end


# shanghai li proposed 2018.4
def cross_seg(df,  # source dataframe
              keyf,  # key field to calculate segment
              cross_field,  # cross field, calculate count for >=keyf_seg & >=vf_seg
              cf_seg_list=(40, 50, 60, 70, 80, 90, 100),  # segment for cross field
              keyf_max=150,
              keyf_min=0
              ):
    """
    交叉表指在某关键列（字段）的分段基础上计算其他有关列（字段）的分段（交叉统计字段）计数值
    :param df: 源数据框
    :param keyf: 关键分段字段
    :param cross_field: 交叉统计计数字段，列+ 最低分数
    :param cf_seg_list: 关键字段分段值列表
    :param keyf_max: max value for keyf
    :param keyf_min: min value for keyf
    :return:
    """
    display_step = 20
    segmodel = SegTable()
    segmodel.set_data(df, keyf)
    segmodel.set_parameters(segmax=keyf_max,
                            segmin=keyf_min)
    segmodel.run()
    dfseg = segmodel.output_data
    dfcount = dfseg[keyf+'_sum'].tail(1).values[0]
    vfseg = {x: [] for x in cf_seg_list}
    vfper = {x: [] for x in cf_seg_list}
    seglen = dfseg['seg'].count()
    for sv, step in zip(dfseg['seg'], range(seglen)):
        if (step % display_step == 0) | (step == seglen-1):
            print('=' * int((step+1)/seglen * 30) + '>>' +
                  f'{float_str((step+1)/seglen, 1, 2)}')
        for vfv in cf_seg_list:
            segcount = df.loc[(df[keyf] >= sv) & (df[cross_field] >= vfv), cross_field].count()
            vfseg[vfv].append(segcount)
            vfper[vfv].append(segcount/dfcount)
    for vs in cf_seg_list:
        dfseg.loc[:, cross_field + '_' + str(vs) + '_sum'] = vfseg[vs]
        dfseg.loc[:, cross_field + '_' + str(vs) + '_percent'] = vfper[vs]
    return dfseg


def float_str(x, d1, d2):
    return ('{:'+str(d1+d2+1)+'.'+str(d2)+'}').format(x)
    # return fs.format(x)


def int_str(x, d):
    # return f'%{d}d' % x
    return ('{:'+str(d)+'d}').format(x)


def round45i(v, dec=0):
    if not isinstance(v, float):
        return v
    u = int(v * 10**dec*10)
    return (int(u/10) + (1 if v > 0 else -1))/10**dec if (abs(u) % 10 >= 5) else int(u/10)/10**dec
