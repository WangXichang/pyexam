# -*- utf-8 -*-
# from 2017-09-16
# version 1.0.2     at 2018-6-24

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


# test SegTable
def test():
    """
    a example for test SegTable
    ---------------------------------------------------------------------------
    expdf = pd.DataFrame({'sf': [1, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, ]})
    seg = SegTable()
    seg.set_data(expdf, expdf.columns.values)
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=3, segalldata=True, display=True)
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
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=3, segalldata=False, display=True)
    seg.run()
    print(seg.output_data)
         seg     sf_count  sf_sum   sf_percent      sf_count3
    5    8         2          2     0.133333          2
    4    7         1          3     0.200000         -1
    3    6         1          4     0.266667         -1
    2    5         1          5     0.333333          3
    1    4         4          9     0.600000         -1
    0    3         6         15     1.000000         10  # excluding 0, 1
    """
    expdf = pd.DataFrame({'sf': [1, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8]})
    print('example dataframe')
    print('='*80)
    print(expdf)

    seg = SegTable()
    seg.set_data(expdf, ['sf'])
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=3, segalldata=True, display=False)
    seg.run()
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)

    # special step, start, no all data
    seg.set_parameters(segstep=3, segstart=8, segmax=8, segmin=3, segalldata=False, display=False)
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
    seg.seglistuse = True
    seg.seglist = [7, 5, 3]
    seg.segsort = 'd'
    seg.segstart = 2
    seg.segmax = 7
    seg.display = False
    seg.run()
    print('\n')
    print('='*80)
    seg.show_parameters()
    print(seg.output_data)
    print('\n--- get result using [(df.sf_list > 0) | (df.seg.isin([segmax, segmin])] ---')
    print(seg.output_data[(seg.output_data.sf_list > 0) | seg.output_data.seg.isin([7, 1])])

    return seg


class SegTable(object):
    """
    * 计算pandas.DataFrame中分数字段的分段人数表
    * segment table for score dataframe
    * version1.01, 2018-06-21
    * from 0917-2017

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(input_data:DataFrame, field_list:list)
        input_data: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        field_list: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离， 分数顺序， 累加分值范围外数据， 关闭计算过程显示
    set_parameters（segmax, segmin, segstep, segsort, segalldata, display）
        segmax: int,  maxvalue for segment, default=150输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。输出分段表中分数段的最小值
        segstep: int, levels for segment value, default=1
                 分段间隔，用于生成n-分段表（五分一段的分段表）
        segStart:int, start seg score to count
                 开始进行分段计算的分值
        seglist: list, used to create set value
                 使用给定的列表产生分段表，列表中为分段点分数值
        seglistuse: bool, use or not use seglist to create seg value
                 是否使用给定列表产生分段值
        segsort: str, 'a' for ascending order or 'd' for descending order, default='d' (seg order on descending)
                 输出结果中分段值得排序方式，descending:从大到小， ascending：从小到大
                 排序模式的设置影响累计数和百分比的意义。
        segalldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        display: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息

    运行计算，产生输出数据, calculate and create output data
    run()

    运行结果：分段计算结果（DataFrame),包含字段seg(分数段), [segfield]_count(本段人数）, [segfield]_sum(累计人数)
              [segfield]_percent(百分数，在顺序中排在其前的人数）
    output_data: dataframe with field 'seg, segfield_count, segfield_sum, segfield_percent'

    应用举例
    example:
        import pyex_seg as sg
        seg = sg.SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='d', segalldata=True, display=True)
        seg.run()
        seg.plot()
        print(seg.output_data.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据segalldata确定是否在设定的区间范围内计算分数值
          segalldata=True时抛弃不在范围内的记录项
          segalldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
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
        self.__segListUse = False
        self.__segStart = 100
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'd'
        self.__segAlldata = True
        self.__display = True
        # result data
        self.__output_dataframe = None
        # run status
        self.__runsuccess = False

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
    def seglistuse(self):
        return self.__segListUse

    @seglistuse.setter
    def seglistuse(self, seglistuse):
        self.__segListUse = seglistuse

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
        return self.__segAlldata

    @segalldata.setter
    def segalldata(self, datamode):
        self.__segAlldata = datamode

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
            seglistuse=None,
            segsort=None,
            segalldata=None,
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
        if isinstance(segalldata, bool):
            set_str += 'set segalldata to {}'.format(segalldata) + '\n'
            self.__segAlldata = segalldata
        if isinstance(display, bool):
            set_str += 'set display to {}'.format(display) + '\n'
            self.__display = display
        if isinstance(segstart, int):
            set_str += 'set segstart to {}'.format(segstart) + '\n'
            self.__segStart = segstart
        if isinstance(seglist, list):
            set_str += 'set seglist to {}'.format(seglist) + '\n'
            self.__segList = seglist
        if isinstance(seglistuse, bool):
            set_str += 'set seglistuse to {}'.format(seglistuse) + '\n'
            self.__segListUse = seglistuse
        if display:
            print(set_str)
        self.__check()
        # self.show_parameters()

    def show_parameters(self):
        print('------ seg parameters ------')
        print('   seg list use:{0}'.format(self.__segListUse, self.__segList))
        print('       seg list:{1}'.format(self.__segListUse, self.__segList))
        print('      max value:{}'.format(self.__segMax))
        print('      min value:{}'.format(self.__segMin))
        print('    start value:{}'.format(self.__segStart))
        print('     step value:{}'.format(self.__segStep))
        print('      sort mode:{}'.format('d (descending)' if self.__segSort in ['d', 'D'] else 'a (ascending)'))
        print('       all data:{}'.format(self.__segAlldata))
        print('        display:{}'.format(self.__display))
        print('-' * 28)

    def help(self):
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
        # #if (self.__segStart < self.__segMin) | (self.__segStart > self.__segMax):
        # #    print('error: segstart({}) is too small or big!'.format(self.__segStart))
        #     return False
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
        if not isinstance(self.__segAlldata, bool):
            print('error: segalldata({}) is not bool type!'.format(self.__segAlldata))
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
        if  self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__output_dataframe = pd.DataFrame({'seg': seglist})
        for f in self.field_list:
            # calculate preliminary group count
            tempdf = self.input_data
            self.input_data.loc[:, f] = tempdf[f].apply(round45i)

            # count seg_count in [segmin, segmax]
            r = self.input_data.groupby(f)[f].count()
            fcount_list = [np.int64(r[x]) if x in r.index else 0 for x in seglist]
            self.__output_dataframe.loc[:, f+'_count'] = fcount_list
            # self.__output_dataframe.loc[:, f + '_count'] = self.__output_dataframe['seg'].\
            #     apply(lambda x: np.int64(r[x]) if x in r.index else 0)
            # if self.__display:
            #     print('segments count finished count ' + f, ' use time:{}'.format(time.clock() - sttime))

            # add outside scope number to segmin, segmax
            if self.__segAlldata:
                self.__output_dataframe.loc[self.__output_dataframe.seg == self.__segMin, f + '_count'] = \
                    r[r.index <= self.__segMin].sum()
                self.__output_dataframe.loc[self.__output_dataframe.seg == self.__segMax, f + '_count'] = \
                    r[r.index >= self.__segMax].sum()

            # calculate cumsum field
            self.__output_dataframe[f + '_sum'] = self.__output_dataframe[f + '_count'].cumsum()
            if self.__segListUse:
                self.__output_dataframe[f + '_list_sum'] = self.__output_dataframe[f + '_count'].cumsum()

            # calculate percent field
            # if self.__display:
            #     print('segments count finished cumsum ' + f, ' use time:{0}'.format(time.clock() - sttime))
            maxsum = max(max(self.output_data[f + '_sum']), 1)     # avoid divided by 0 in percent computing
            self.__output_dataframe[f + '_percent'] = \
                self.__output_dataframe[f + '_sum'].apply(lambda x: x / maxsum)
            if self.__display:
                print('segments count finished[' + f, '], used time:{}'.format(time.clock() - sttime))

            # special seg step
            if self.__segStep > 1:
                self.__run_special_step(f)

            # use seglist
            if self.__segListUse:
                if len(self.__segList) > 0:
                    self.__run_seg_list(f)

        if self.__display:
            print('segments count total consumed time:{}'.format(time.clock()-sttime))
            print('=== end')
        self.__runsuccess = True
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
            if self.__segAlldata | (minpoint <= curseg <= maxpoint):
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
                if self.__segAlldata:
                    self.__output_dataframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1

    def plot(self):
        if not self.__runsuccess:
            if self.__display:
                print('result is not created, please run!')
            return
        legendlist = []
        step = 0
        for sf in self.field_list:
            step += 1
            legendlist.append(sf)
            plt.figure('seg table figure({})'.
                       format('Descending' if self.__segSort in 'aA' else 'Ascending'))
            plt.subplot(221)
            plt.hist(self.input_data[sf], 20)
            plt.title('raw data histogram')
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.subplot(222)
            plt.plot(self.output_data.seg, self.output_data[sf+'_count'])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.title('seg -- count')
            plt.xlim([self.__segMin, self.__segMax])
            plt.subplot(223)
            plt.plot(self.output_data.seg, self.output_data[sf + '_sum'])
            plt.title('seg -- cumsum')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.subplot(224)
            plt.plot(self.output_data.seg, self.output_data[sf + '_percent'])
            plt.title('seg -- percent')
            plt.xlim([self.__segMin, self.__segMax])
            if step == len(self.field_list):
                plt.legend(legendlist)
            plt.show()

    # SegTable class end


# shanghai li proposed 2018.4
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
    dfseg = segmodel.output_data
    dfcount = dfseg[keyf+'_sum'].tail(1).values[0]
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
        dfseg[vf + str(vs) + '_sum'] = vfseg[vs]
        dfseg[vf + str(vs) + '_percent'] = vfper[vs]
    return dfseg


def float_str(x, d1, d2):
    d1 = d1 + d2 + 1
    # return f'%{d1}.{d2}f' % x
    fstr = '{:'+str(d1)+'.'+str(d2)+'}'
    return fstr.format(x)


def int_str(x, d):
    # return f'%{d}d' % x
    fstr = '{:'+str(d)+'d}'
    return fstr.format(x)

def round45i(v:float, dec=0):
    u = int(v * 10**dec*10)
    return (int(u/10) + (1 if v >0 else -1))/10**dec if (abs(u) % 10 >= 5) else int(u/10)/10**dec
