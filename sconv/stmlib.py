# coding: utf-8


import time
import os
import re
import numbers
import configparser as cfp
from collections import namedtuple
import logging
from logging import handlers


import numpy as np
import scipy.stats as sts
import pandas as pd
import matplotlib.pyplot as plot


# call SegTable.run() return instance of SegTable
def get_segtable(
            df: pd.DataFrame,
            cols: list,
            segmax=100,
            segmin=0,
            segsort='d',
            segstep=1,
            display=False,
            usealldata=False
            ):
    seg = SegTable()
    seg.set_data(
        df=df,
        cols=cols
    )
    seg.set_para(
        segmax=segmax,
        segmin=segmin,
        segstep=segstep,
        segsort='a' if segsort in ['a', 'ascending'] else 'd',
        display=display,
        useseglist=usealldata
    )
    seg.run()
    return seg


class SegTable(object):
    """
    * 分数分段及百分位表模型
    * model for score segment-percentile table
    * from 09-17-2017
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    # version 1.0.1 2018-09-24

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(df:DataFrame, fs:list)
        df: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        fs: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离，分段开始值，分数顺序，指定分段值列表， 使用指定分段列表，使用所有数据， 关闭计算过程显示信息
    set_para（segmax, segmin, segstep, segstart, segsort, seglist, useseglist, usealldata, display）
        segmax: int, maxvalue for segment, default=150
                输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。
                输出分段表中分数段的最小值
        segstep: int, grades for segment value, default=1
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
    outdf: 输出分段数据
            seg: seg value
        [field]: field name in fs
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
        seg = SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_para(segmax=100, segmin=1, segstep=1, segsort='d', usealldata=True, display=True)
        seg.run()
        print(seg.outdf.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          fs type is digit, for example: int or float

        3)可以单独设置数据(df),字段列表（fs),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.col = ['score_1', 'score_2'];
              seg.segmax = 120
          重新设置后需要运行才能更新输出数据ouput_data, 即调用run()
          便于在计算期间调整模型。
          by usting property mode, rawdata, scorefields, para can be setted individually
        4) 当设置大于1分的分段分值X时， 会在结果DataFrame中生成一个字段[segfiled]_countX，改字段中不需要计算的分段
          值设为-1。
          when segstep > 1, will create field [segfield]_countX, X=str(segstep), no used value set to -1 in this field
    """

    def __init__(self):
        # raw data
        self.__dfframe = None
        self.__cols = []

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
        self.__percent_decimal = 10

        # result data
        self.__outdfframe = None

        # run status
        self.__run_completed = False

    @property
    def outdf(self):
        return self.__outdfframe

    @property
    def df(self):
        return self.__dfframe

    @df.setter
    def df(self, df):
        self.__dfframe = df

    @property
    def cols(self):
        return self.__cols

    @cols.setter
    def cols(self, cols):
        self.__cols = cols

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

    def set_data(self, df, cols=None):
        self.__dfframe = df
        if isinstance(cols, str):
            cols = [cols]
        if (not isinstance(cols, list)) & (not isinstance(cols, tuple)):
            self.__cols = []
        else:
            self.__cols = cols
        self.__check()

    def set_para(
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
        if segmax is not None:
            self.__segMax = segmax
            set_str += 'set segmax to {}'.format(segmax) + '\n'
        if segmin is not None:
            self.__segMin = segmin
            set_str += 'set segmin to {}'.format(segmin) + '\n'
        if segstep is not None:
            self.__segStep = segstep
            set_str += 'set segstep to {}'.format(segstep) + '\n'
        if segstart is not None:
            set_str += 'set segstart to {}'.format(segstart) + '\n'
            self.__segStart = segstart
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
            self.show_para()

    def show_para(self):
        print('------ seg para ------')
        print('    use seglist:{}'.format(self.__useseglist))
        print('        seglist:{}'.format(self.__segList))
        print('       maxvalue:{}'.format(self.__segMax))
        print('       minvalue:{}'.format(self.__segMin))
        print('       segstart:{}'.format(self.__segStart))
        print('        segstep:{}'.format(self.__segStep))
        print('        segsort:{}'.format('d (descending)' if self.__segSort in ['d', 'D'] else 'a (ascending)'))
        print('     usealldata:{}'.format(self.__usealldata))
        print('        display:{}'.format(self.__display))
        print('-' * 28)

    def help_doc(self):
        print(self.__doc__)

    def __check(self):
        if isinstance(self.__dfframe, pd.Series):
            self.__dfframe = pd.DataFrame(self.__dfframe)
        if not isinstance(self.__dfframe, pd.DataFrame):
            print('error: raw score data is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('error: segmax({}) is not greater than segmin({})!'.format(self.__segMax, self.__segMin))
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('error: segstep({}) is too small or big!'.format(self.__segStep))
            return False
        if not isinstance(self.cols, list) and not isinstance(self.cols, tuple):
            print(self.__cols)
            if isinstance(self.__cols, str):
                self.__cols = [self.__cols]
            else:
                print('error: segfields type=({})!'.format(type(self.__cols)))
                return False

        for f in self.__cols:
            if f not in self.df.columns:
                print("error: field('{}') is not in df fields({})".
                      format(f, self.df.columns.values))
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
            print('---seg calculation start---')
        if self.segstep >= 1:
            seglist = [x for x in range(int(self.__segMin), int(self.__segMax + 1))]
        else:
            _segnum = (self.__segMax - self.__segMin + self.__segStep)/self.__segStep
            seglist = [self.__segMin+i*self.__segStep for i in range(int(_segnum))
                       if self.__segMin+i*self.__segStep < self.__segMax] + \
                      [self.__segMax]
        if self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__outdfframe = pd.DataFrame({'seg': seglist})
        outdf = self.__outdfframe
        for f in self.__cols:
            # calculate preliminary group count
            tempdf = self.df
            tempdf.loc[:, f] = tempdf[f].apply(round45)

            # count seg_count in [segmin, segmax]
            r = tempdf.groupby(f)[f].count()
            # fcount_list = [np.int64(r[x]) if x in r.index else 0 for x in seglist]
            outdf.loc[:, f+'_count'] = [np.int64(r[x]) if x in r.index else 0 for x in seglist]
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
                outdf[f + '_sum'].apply(lambda x: round45(x / maxsum, self.__percent_decimal))
            if self.__display:
                print('segments count finished[' + f, '], used time:{}'.format(time.clock() - sttime))

            # self.__outdfframe = outdf.copy()
            # special seg step
            if self.__segStep > 1:
                self.__run_special_step(f)

            # use seglist
            if self.__useseglist:
                if len(self.__segList) > 0:
                    self.__run_seg_list(f)

        if self.__display:
            print('segments count total consumed time:{}'.format(time.clock()-sttime))
            print('---seg calculation end---')
        self.__run_completed = True
        self.__outdfframe = outdf
        return

    def __run_special_step(self, field: str):
        """
        processing count for step > 1
        :param field: for seg stepx
        :return: field_countx in outdf
        """
        f = field
        segcountname = f + '_count{0}'.format(self.__segStep)
        self.__outdfframe[segcountname] = np.int64(-1)
        curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        curpoint = self.__segStart
        if self.__segSort.lower() == 'd':
            while curpoint+curstep > self.__segMax:
                curpoint += curstep
        else:
            while curpoint+curstep < self.__segMin:
                curpoint += curstep
        cum = 0
        for index, row in self.__outdfframe.iterrows():
            cum += row[f + '_count']
            curseg = np.int64(row['seg'])
            if curseg in [self.__segMax, self.__segMin]:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if (self.__segStart <= self.__segMin) | (self.__segStart >= self.__segMax):
                    curpoint += curstep
                continue
            if curseg in [self.__segStart, curpoint]:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
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
        self.__outdfframe[segcountname] = np.int64(-1)
        segpoint = sorted(self.__segList) \
            if self.__segSort.lower() == 'a' \
            else sorted(self.__segList)[::-1]
        # curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        # curpoint = self.__segStart
        cum = 0
        pos = 0
        curpoint = segpoint[pos]
        rownum = len(self.__outdfframe)
        cur_row = 0
        lastindex = 0
        maxpoint = max(self.__segList)
        minpoint = min(self.__segList)
        list_sum = 0
        self.__outdfframe.loc[:, f+'_list_sum'] = 0
        for index, row in self.__outdfframe.iterrows():
            curseg = np.int64(row['seg'])
            # cumsum
            if self.__usealldata | (minpoint <= curseg <= maxpoint):
                cum += row[f + '_count']
                list_sum += row[f+'_count']
                self.__outdfframe.loc[index, f+'_list_sum'] = np.int64(list_sum)
            # set to seg count, only set seg in seglist
            if curseg == curpoint:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if pos < len(segpoint)-1:
                    pos += 1
                    curpoint = segpoint[pos]
                else:
                    lastindex = index
            elif cur_row == rownum:
                if self.__usealldata:
                    self.__outdfframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1
# SegTable class end


def round45(number, digits=0):
    int_len = len(str(int(abs(number))))
    if int_len + abs(digits) <= 16:
        err_ = (1 if number >= 0 else -1)*10**-(16-int_len)
        if digits > 0:
            return round(number + err_, digits) + err_
        else:
            return int(round(number + err_, digits))
    else:
        raise NotImplemented


def timer_wrapper(fun):

    def dec_fun(*args, **kwargs):
        st = time.time()
        print('process start: {}'.format(fun))
        result = fun(*args, **kwargs)
        print('process[{}] elapsed time: {:.3f}'.format(fun, time.time() - st))
        return result

    return dec_fun


def get_norm_point_pdf(start=100,
                       end=900,
                       loc=500,
                       std=100,
                       step=1,
                       add_cutoff=True,
                       mode='middle'
                       ):
    point_list = list(range(start, end+1, step))
    ratio_pdf = [0 for _ in range(start, end+1, step)]

    i = 0
    for p in range(start, end+1, step):
        z = (p - loc) / std
        if mode == 'middle':
            ratio_pdf[i] = sts.norm.cdf(z + step/2/std) - sts.norm.cdf(z - step/2/std)
        elif mode == 'left':
            if i > 0:
                ratio_pdf[i] = sts.norm.cdf(z) - sts.norm.cdf(z - step/std)
            else:
                ratio_pdf[i] = sts.norm.cdf(z)
        elif mode == 'right':
            if i < end:
                ratio_pdf[i] = sts.norm.cdf(z + step/std) - sts.norm.cdf(z)
            else:
                ratio_pdf[i] = 1 - sts.norm.cdf(z)
        else:
            raise ValueError
        i += 1
    if mode == 'middle':
        cutoff = sts.norm.cdf((start - loc - step / 2) / std)
    else:
        cutoff = sts.norm.cdf((start - loc) / std)
    if add_cutoff:
        if mode in ['left', 'middle']:
            ratio_pdf[0] = ratio_pdf[0] + cutoff
        ratio_pdf[-1] = 1 - sum(ratio_pdf[0:-1])
    ratio_cumu = [sum(ratio_pdf[:i+1]) for i in range(len(ratio_pdf))]
    result = namedtuple('R', ['pdf', 'cdf', 'points', 'cutoff'])
    return result(ratio_pdf, ratio_cumu, point_list, cutoff)


def get_norm_section_pdf(
                    start=21,
                    end=100,
                    section_num=8,
                    std_num=2.6,
                    add_cutoff=True,
                    model_type='plt',
                    ratio_coeff=1,      # 1, or 100
                    sort_order='d',
                    mode='left',        # left, middle, right
                    ):
    """
    create a ratio table from norm distribution
        with range=[start, end]
             section = (start, start+j*step), ...
                       j in range(start, end, step)
                       step = (end - start)/section_num
              cutoff = norm.cdf((mean - start)/std_num)
    return result: pdf, cdf, cutoff_err
    set first and end seg to tail ratio from norm table
    can be used to test std from seg ratio table
    for example,
       get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.9508)
       [0.03000, 0.07495, 0.16040, 0.23465, ..., 0.03000],
       get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.6065)
       [0.02729, 0.07272, 0.16083, 0.23916, ..., 0.02729],
       it means that std==15.95 is fitting ratio 0.03,0.07 in the table

    :param start:   start value
    :param end:     end value
    :param section_num: section number
    :param std_num:     length from 0 to max equal to std_num*std, i.e. std = (end-start)/2/std_num
    :param add_cutoff: bool, if adding cutoff cdf() at edge point
                       i.e. cdf(-std_num), cdf(-4) = 3.167124183311986e-05, cdf(-2.5098)=0.029894254950869625
    :param model_type: str, 'plt' or 'ppt'
    :param ratio_coeff: int, 1 for 0-1, 100 for 0-100, amplify ratio to ratio*ratio_coeff scale
    :param sort_order: str, order for points
    :param mode: str, how to calculate pdf for each discret point
    :return: namedtuple('result', ('section':((),...), 'pdf': (), 'cdf': (), 'cutoff': float, 'add_cutoff': bool))
    """
    _mean, _std = (end + start) / 2, (end - start) / 2 / std_num
    section_point_list = np.linspace(start, end, section_num + 1)
    section_len = (end - start) / section_num
    cutoff = sts.norm.cdf((start - _mean) / _std)
    pdf_table = [0]
    cdf_table = [0]
    last_pos = (start - _mean) / _std
    _cdf = 0
    for i, pos in enumerate(section_point_list[1:]):
        _zvalue = (pos - _mean) / _std
        if mode == 'left':
            this_section_pdf = sts.norm.cdf(_zvalue) - sts.norm.cdf(last_pos)
        elif mode == 'right':
            this_section_pdf = sts.norm.cdf(_zvalue) - sts.norm.cdf(last_pos)
        else:
            _step = section_len / _std
            this_section_pdf = sts.norm.cdf(_zvalue+_step/2) - sts.norm.cdf(_zvalue-_step/2)
        # if (i == 0) and add_cutoff:
        #     this_section_pdf += cutoff
        pdf_table.append(this_section_pdf)
        cdf_table.append(this_section_pdf + _cdf)
        last_pos = _zvalue
        _cdf += this_section_pdf

    # add cutoff
    if add_cutoff:
        pdf_table[1] += cutoff
        pdf_table[-1] += cutoff
        cdf_table[-1] = 1

    # make section
    if model_type == 'plt':
        section_list = [(x, y) if i == 0 else (x + 1, y)
                        for i, (x, y) in enumerate(zip(section_point_list[:-1], section_point_list[1:]))]
    else:
        section_list = [(x, x) for x in section_point_list]

    # amplify
    if ratio_coeff != 1:
        pdf_table = [x * ratio_coeff for x in pdf_table]

    # sort
    if sort_order in ['d', 'descending']:
        section_list = sorted(section_list, key=(lambda x: -x[0]))

    result = namedtuple('Result', ('section', 'pdf', 'cdf', 'point', 'cutoff', 'add_cutoff'))
    r = result(tuple(section_list),
               tuple(pdf_table),
               tuple(cdf_table),
               section_point_list,
               cutoff,
               add_cutoff)
    return r


def isfilename(fstr):
    if isinstance(fstr, str):
        _invalid_file_char = "[/*?:<>|\"\'\\\\]"
        if len(fstr) > 0:
            for c in fstr:
                if c in _invalid_file_char:
                    # print('error logname: {} is invalid char, not allowed in: \"{}\"'.
                    #       format(c, _invalid_file_char))
                    return False
            return True
    return False


def read_config_file(conf_name):
    mcfg = dict()
    cfper = dict()

    if os.path.isfile(conf_name):
        cfper = cfp.ConfigParser()
        cfper.read(conf_name, encoding='utf-8')

    if 'model_in' in cfper.keys():
        if 'name' in cfper['model_in']:
            mcfg.update({'model_name': remove_annotation(cfper['model_in']['name'])})
            mcfg.update({'model_in_check': True})
        else:
            mcfg.update({'model_in_check': False})

    if not mcfg['model_in_check']:
        model_list = ['name', 'type', 'ratio', 'section', 'desc']
        new_set = False
        if 'model_new' in cfper.keys():
            for k in model_list:
                if k in cfper['model_new'].keys():
                    _s = remove_annotation(cfper['model_new'][k])
                    if k.lower() == 'ratio':
                        s = _s.split(',')
                        s = [float(x) for x in s]
                    elif k.lower() == 'section':
                        s = re.findall('[0-9]+', _s)
                        s = [(float(x), float(y)) for x, y in zip(s[0::2], s[1::2])]
                    else:
                        s = _s
                    mcfg.update({'model_new_' + k: s})
                    new_set = True
                else:
                    print('config error: model para incomplete in model_new: {}'.format(k))
        if new_set:
            _ch = Checker.check_model_para(
                model_type=mcfg['model_new_type'],
                model_ratio=mcfg['model_new_ratio'],
                model_section=mcfg['model_new_section'],
                model_desc=mcfg['model_new_desc']
                )
            mcfg.update({'model_new_check': _ch})
            mcfg.update({'model_new_set': True})
        else:
            mcfg.update({'model_new_set': False})

    if 'data' in cfper.keys():
        df = None
        if 'df' in cfper['data']:
            dffile = remove_annotation(cfper['data']['df'])
            mcfg.update({'df:filename': dffile})
            if os.path.isfile(dffile):
                try:
                    df = pd.read_csv(dffile)
                    mcfg.update({'df': df})
                except IOError:
                    print('data read error!')
            else:
                print('invalid data file name: df = {}'.format(dffile))
        else:
            print('no df in config file!')
        mcfg.update({'df': df})
        cols = None
        if 'cols' in cfper['data']:
            _cols = remove_annotation(cfper['data']['cols'])
            _cols = _cols.split()
            _cols = [x.replace(',', '') for x in _cols]
            _cols = [x.replace(';', '') for x in _cols]
            _cols = [x.strip() for x in _cols]
            cols = _cols
        else:
            print('no cols in config file!')
        mcfg.update({'cols': cols})

    if 'value' in cfper.keys():
        for _para in cfper['value']:
            mcfg.update({_para: remove_annotation(cfper['value'][_para])})

        if 'value_raw_score_min' in mcfg.keys():
            mcfg['value_raw_score_min'] = int(mcfg['value_raw_score_min'])
        else:
            mcfg.update({'value_raw_score_min': 0})

        if 'value_raw_score_max' in mcfg.keys():
            mcfg['value_raw_score_max'] = int(mcfg['value_raw_score_max'])
        else:
            mcfg.update({'value_raw_score_max': 100})

        if 'value_raw_score_step' in mcfg.keys():
            mcfg['value_raw_score_step'] = int(mcfg['value_raw_score_step'])
        else:
            mcfg.update({'value_raw_score_step': 1})

        if 'value_out_score_decimals' in mcfg.keys():
            mcfg['value_out_score_decimals'] = int(mcfg['value_out_score_decimals'])
        else:
            mcfg.update({'value_out_score_decimals': 0})

        if 'value_tiny_value' in mcfg.keys():
            mcfg['value_tiny_value'] = eval(mcfg['value_tiny_value'])
        else:
            mcfg['value_tiny_value'] = 10**-10

    if 'task' in cfper.keys():
        for _para in cfper['task']:
            mcfg.update({_para: remove_annotation(cfper['task'][_para])})

        if 'logname' in mcfg.keys():
            s = mcfg['logname']
            s = s.replace("'", "")
            s = s.replace('"', '')
            mcfg['logname'] = s
        else:
            mcfg.update({'logname': ''})

        # set log para bool items
        log_bool_list = ['logdisp', 'logfile', 'verify', 'logdata']
        default_d = {'logdisp': True,
                     'logfile': False,
                     'logdata': False,
                     'verify': False,
                     }
        for ks in log_bool_list:
            if ks in mcfg.keys():
                if mcfg[ks].lower() in ['true', '1']:
                    mcfg[ks] = True
                else:
                    mcfg[ks] = False
            else:
                mcfg.update({ks: default_d[ks]})

        # set loglevel
        if 'loglevel' in mcfg.keys():
            _set = False
            ms = mcfg['loglevel'].lower()
            for s in ['debug', 'info', 'warnning', 'error', 'critical']:
                if s in ms:
                    mcfg['loglevel'] = s
                    _set = True
                    break
            if not _set:
                # print('set loglevel error, invalid value: {}!'.format(ms))
                mcfg['loglevel'] = 'info'
        else:
            mcfg.update({'loglevel': 'info'})

    if 'mode' in cfper.keys():
        for _mode in cfper['mode']:
            _mode_str = remove_annotation(cfper['mode'][_mode])
            mcfg.update({_mode: _mode_str})

    # set defaul mode
    defa_mode = {'mode_ratio_prox': 'upper_min',
                 'mode_score_order': 'd',
                 'mode_score_prox': 'upper_min',
                 'mode_ratio_cumu': 'no',
                 'mode_endpoint_first': 'real',
                 'mode_endpoint_start': 'step',
                 'mode_endpoint_last': 'real',
                 'mode_section_shrink': 'to_max',
                 'mode_section_lost': 'real'}
    for _mode in defa_mode:
        if _mode not in mcfg:
            mcfg.update({_mode: defa_mode[_mode]})

    return mcfg


def remove_annotation(s):
    p = s.find('#')
    rs = s.lower()
    if p >= 0:
        rs = s[0:p].strip().lower()
    return rs


def make_config_file(filename):

    # verify is used to test only in task
    # verify = 0  # use dual algorithm to verify result or not
    template = \
        """
        [task]
        logname = test                      # 任务名称，用于生成日志: logname_model_year_month_day.log
        logdisp = True                      # 是否显示计算过程信息 message to consol
        logfile = False                     # 是否将计算过程信息写入日志 message to log file
        logdata = False                     # 是否保存结果数据：[logname]_df_outscore/maptable_[modelname]_[time].csv

        
        [data]
        df = rawscore.csv                   # 原始分数文件名, csv文件格式，第一行表示列名
        cols = km1, km2                     # 分数列名, 以逗号分隔，数值类型（整数或浮点数）

        
        [model_in]
        name = shandong                     # 模型名称,不设置该名称，则使用 model_new 的模型设置
        
        
        [value]
        value_raw_score_min = 0             # 原始分数卷面最小值 min score for raw score
        value_raw_score_max = 100           # 原始分数卷面最大值 max score for raw score
        value_raw_score_step = 1            # 原始分数的分值间隔（步长）raw score step
        value_out_score_decimals = 0        # 转换分数保留小数位 decimal digits for out score
        value_tiny_value = 10**-12          # 计算使用的微小值，小于该值的误差被忽略 smallest value for precision

                
        [mode]
        mode_score_order = d                  # 排序策略：d, a
        mode_score_prox = upper_min           # 分值逼近策略：upper_min, lower_max, near_max, near_min
        mode_ratio_prox = upper_min           # 比例逼近策略：upper_min, lower_max, near_max, near_min
        mode_ratio_cumu = no                  # 比例累计策略：no, yes
        mode_endpoint_first = real            # 第一端点策略：real, defined
        mode_endpoint_start = step            # 开始端点策略：step, share
        mode_endpoint_last = real             # 最后端点策略：real, defined
        mode_section_shrink = to_max          # 区间退化策略：to_max, to_min, to_mean (映射到最大、最小、平均值)
        mode_section_lost = real              # 区间消失策略：real, zip

        
        [model_new]
        name = model-001                      # 自定义模型名称: 使用合法文件名字符, valid char used in file name
        type = plt                            # 自定义模型类型: plt, ppt, pgt
        section = (150, 131), (130, 111), (110, 91), (90, 71), (70, 51)，（50，30）       # 转换分数区间（降序） out score section (descending)
        ratio =   2, 13, 35, 35, 13, 2                                                  # 原始分数等级区间划分比例（百分数0-100），和等于100, ratio(%), sum==100
        """

    if isfilename(filename):
        with open(filename, 'a', encoding='utf8') as fp:
            ms = template.strip().split('\n')
            for ss in ms:
                fp.write(ss.strip() + '\n')
        return True
    else:
        return False


class Checker:

    MODEL_TYPE_LIST = ('plt', 'ppt', 'pgt')

    @staticmethod
    def check_run(
            model_name='shandong',
            df=None,
            cols=None,
            mode_ratio_prox='upper_min',
            mode_ratio_cumu='no',
            mode_score_prox='upper_min',
            mode_score_order='d',
            mode_endpoint_first='real',
            mode_endpoint_start='step',
            mode_endpoint_last='real',
            mode_section_shrink='map_to_max',
            mode_section_lost='real',
            raw_score_range=(0, 100),
            out_score_decimal_digits=0,
            logger=None,
            models=None
            ):

        if logger is None:
            logger = get_logger('check')
            logger.logging_consol = True
            logger.logging_file = False

        # check model name
        if model_name not in models.Models.keys():
            logger.loginfo('error model name: {} !'.format(model_name))
            return False

        # check input data: DataFrame
        if not Checker.check_df_cols(df, cols, raw_score_range, logger):
            return False

        # check strategy
        if not Checker.check_strategy(
                mode_ratio_prox=mode_ratio_prox,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_score_prox=mode_score_prox,
                mode_score_order=mode_score_order,
                mode_endpoint_first=mode_endpoint_first,
                mode_endpoint_start=mode_endpoint_start,
                mode_endpoint_last=mode_endpoint_last,
                mode_section_shrink=mode_section_shrink,
                mode_section_lost=mode_section_lost,
                logger=logger,
                models=models,
        ):
            return False

        if out_score_decimal_digits < 0 or out_score_decimal_digits > 10:
            logger.logger.info('warning: decimal digits={} set may error!'.format(out_score_decimal_digits))

        return True

    @staticmethod
    def check_model(model_name, model_lib=None, logger=None):
        if model_name in model_lib.keys():
            if not Checker.check_model_para(
                model_lib[model_name].type,
                model_lib[model_name].ratio,
                model_lib[model_name].section,
                model_lib[model_name].desc,
                logger=logger
            ):
                return False
        else:
            return False
        return True

    @staticmethod
    def check_model_para(
                    model_type='plt',
                    model_ratio=None,
                    model_section=None,
                    model_desc='',
                    logger=None,
                    ):

        # set logger
        if logger is None:
            logger = get_logger('check')
            logger.logging_consol = True
            logger.logging_file = False

        # check type
        if model_type not in Checker.MODEL_TYPE_LIST:
            logger.loginfo('error type: valid type must be in {}'.format(model_type, ['ppt', 'plt', 'pgt']))
            return False

        # check ratio
        if model_type == 'pgt':
            if len(model_ratio) == 0:
                logger.loginfo('error ratio: length == 0 in model={}!'.format(model_type))
                return False
            if model_ratio[0] < 0 or model_ratio[0] > 100:
                logger.loginfo('error ratio: in type=tai, ratrio[0]={} '
                               'must be range(0, 101) as the percent of top score ratio!'.format(model_ratio[0]))
                return False
        else:
            if len(model_ratio) != len(model_section):
                logger.loginfo('error length: the length of ratio group is not same as section group length !')
                return False
            if abs(sum(model_ratio) - 100) > 10**-12:
                logger.loginfo('error ratio: the sum of ratio must be 100, real sum={}!'.format(sum(model_ratio)))
                return False

        # check section
        for s in model_section:
            if len(s) > 2:
                logger.loginfo('error section: section must have 2 endpoints, real value: {}'.format(s))
                return False
            if s[0] < s[1]:
                logger.loginfo('error order: section endpoint order must be from large to small, '
                               'there: p1({}) < p2({})'.format(s[0], s[1]))
                return False
        if model_type in ['ppt', 'pgt']:
            if not all([x == y for x, y in model_section]):
                logger.loginfo('error section: ppt section, two endpoints must be same value!')
                return False

        # check desc
        if not isinstance(model_desc, str):
            logger.loginfo('error desc: model desc(ription) must be str, but real type={}'.format(type(model_desc)))

        return True

    @staticmethod
    def check_strategy(
            mode_ratio_prox='upper_min',
            mode_ratio_cumu='no',
            mode_score_prox='upper_min',
            mode_score_order='descending',
            mode_endpoint_first='real',
            mode_endpoint_start='step',
            mode_endpoint_last='real',
            mode_section_shrink='map_to_max',
            mode_section_lost='ignore',
            logger=None,
            models=None,
            ):

        if logger is None:
            logger = get_logger('check')
            logger.logging_consol = True
            logger.logging_file = False

        st = {
              'mode_ratio_prox': mode_ratio_prox,
              'mode_ratio_cumu': mode_ratio_cumu,
              'mode_score_prox': mode_score_prox,
              'mode_score_order': mode_score_order,
              'mode_endpoint_first': mode_endpoint_first,
              'mode_endpoint_start': mode_endpoint_start,
              'mode_endpoint_last': mode_endpoint_last,
              'mode_section_shrink': mode_section_shrink,
              'mode_section_lost': mode_section_lost,
              }
        for sk in st.keys():
            if sk in models.Strategy.keys():
                if not st[sk] in models.Strategy[sk]:
                    logger.loginfo('error mode: {}={} not in {}'.format(sk, st[sk], models.Strategy[sk]))
                    return False
            else:
                logger.loginfo('error mode: {} is not in Strategy-dict!'.format(sk))
                return False
        return True

    @staticmethod
    def check_df_cols(df=None, cols=None, raw_score_range=None, logger=None):
        if logger is None:
            logger = get_logger('check')
            logger.logging_consol = True
            logger.logging_file = False
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, pd.Series):
                logger.loginfo('warning: df is pandas.Series!')
                return False
            else:
                logger.loginfo('error data: df(type={}) is not pandas.DataFrame!'.format(type(df)))
                return False
        if len(df) == 0:
            logger.loginfo('error data: df is empty!')
            return False
        if type(cols) not in (list, tuple):
            logger.loginfo('error type: cols must be list or tuple, real type is {}!'.format(type(cols)))
            return False
        for col in cols:
            if type(col) is not str:
                logger.log('error col: {} is not str!'.format(col), 'error')
                return False
            else:
                if col not in df.columns:
                    logger.log('error col: {} is not in df.columns!'.format(col), 'error')
                    return False
                if not isinstance(df.iloc[0][col], numbers.Real):
                    logger.log('type error: column[{}] not Number type!'.format(col), 'error')
                    return False
                _min = df[col].min()
                if _min < min(raw_score_range):
                    logger.loginfo('warning: some scores in col={} not in raw score range:{}'.
                                   format(_min, raw_score_range))
                _max = df[col].max()
                if _max > max(raw_score_range):
                    logger.loginfo('warning: some scores in col={} not in raw score range:{}'.
                                   format(_max, raw_score_range))
        return True


def get_logger(model_name, logname=None):
    gmt = time.localtime()
    if not isinstance(logname, str):
        task_str = 'sconv'
    else:
        if len(logname) == 0:
            task_str = 'sconv'
        else:
            task_str = logname
    log_file = \
        task_str + '_log_' + \
        model_name + '_' + \
        str(gmt.tm_year) + '_' + \
        str(gmt.tm_mon) + '_' + \
        str(gmt.tm_mday) + \
        '.log'
    stmlog = Logger(log_file, level='info')
    return stmlog


class Logger(object):
    level_relations = {
        'debug':    logging.DEBUG,
        'info':     logging.INFO,
        'warn':     logging.WARNING,
        'error':    logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
                 filename='_test.log',
                 level='info',
                 when='D',
                 back_count=3,
                 ):

        # output
        self.logging_file = False
        self.logging_consol = False

        # para
        self.filename = filename
        self.level = level
        self.when = when
        self.back_count = 3 if back_count is not int else back_count
        self.format = '   %(message)s'
        self.when = when
        # self.format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        # logger
        self.logger = logging.getLogger(self.filename)              # 设置日志文件名
        self.logger.setLevel(self.level_relations.get(self.level))  # 设置日志级别
        self.logger_format = logging.Formatter(self.format)         # 设置日志格式

        # handlers
        self.stream_handler = None
        self.rotating_file_handler = None
        self.set_handlers(self.logger_format)

    def set_level(self, level):
        self.level = level
        self.logger.setLevel(self.level_relations.get(self.level))

    def log(self, ms='', level='info'):
        # self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别error
        self.logger.handlers = []
        if self.logging_consol:
            self.logger.addHandler(self.stream_handler)
        if self.logging_file:
            self.logger.addHandler(self.rotating_file_handler)
        if level.lower().strip() == 'error':
            self.logger.error(ms)
        elif level.lower().strip() == 'critical':
            self.logger.critical(ms)
        elif level.lower().strip() == 'info':
            self.logger.info(ms)
        elif level.lower().strip() == 'warning':
            self.logger.warning(ms)
        else:
            self.logger.debug(ms)
        self.logger.handlers = []
        # self.logger.setLevel(self.level_relations.get(self.level))  # 恢复日志级别

    def loginfo(self, ms=''):
        self.logger.handlers = []
        if self.logging_consol:
            self.logger.addHandler(self.stream_handler)
        if self.logging_file:
            self.logger.addHandler(self.rotating_file_handler)
        self.logger.info(ms)
        self.logger.handlers = []

    def loginfo_start(self, ms=''):
        first_logger_format = logging.Formatter(
            '='*120 + '\n[%(message)s] start at [%(asctime)s]\n' + '-'*120)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        self.set_handlers(self.logger_format)

    def loginfo_end(self, ms=''):
        first_logger_format = logging.Formatter(
            '-'*120 + '\n[%(message)s]   end at [%(asctime)s]\n' + '='*120)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        # self.logger = logging.getLogger('_test.log')
        # self.set_handlers(self.logger_format)
        logging.shutdown()

    def set_handlers(self, log_format):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(log_format)
        self.rotating_file_handler = handlers.TimedRotatingFileHandler(
                    filename=self.filename,
                    when=self.when,
                    backupCount=self.back_count,
                    encoding='utf-8'
                )
        self.rotating_file_handler.setFormatter(log_format)


def get_norm_data(mu=50, std=15, size=60000, maxvalue=100, minvalue=0, decimals=0):
    return TestData(mu, std, size, maxscore=maxvalue, minscore=minvalue, decimals=decimals)()


# test dataset
class TestData:
    """
    生成具有正态分布的数据，类型为 pandas.DataFrame, 列名为 sv
    create a score dataframe with fields 'score', used to test some application
    :__init__:parameter
        mean: 均值， std:标准差， max:最大值， min:最小值， size:行数
    :df
        DataFrame, columns = {'ksh', 'km1', 'km2'}
    """
    def __init__(self, mu=60, std=18, size=60000, maxscore=100, minscore=0, decimals=0):
        self.df = None
        self.df_mean = mu
        self.df_max = maxscore
        self.df_min = minscore
        self.df_std = std
        self.df_size = size
        self.decimals = decimals
        # only implement norm distribution
        self.dist = 'norm'
        self.__make_data()

    def __make_data(self):
        import random
        ssno = [str(i).zfill(2) for i in range(1, 17)]
        xqno = [str(i).zfill(2) for i in range(2, 90)]
        klno = ['1', '3', '4']     # 1: regular,  3: art,   4: physical educatio
        self.df = pd.DataFrame({
            'ksh': ['37'+random.sample(ssno, 1)[0] +
                    random.sample(xqno, 1)[0] +
                    random.sample(klno, 1)[0] +
                    str(x).zfill(7)
                    for x in range(1, self.df_size+1)],
            'km1': self.get_score(),
            'km2': self.get_score(),
        })

    def get_score(self):

        def myround(x):
            if self.decimals == 0:
                return int(x)
            else:
                return round(x, ndigits=self.decimals)

        if self.dist == 'norm':
            norm_list = sts.norm.rvs(loc=self.df_mean, scale=self.df_std, size=self.df_size)
            norm_list = np.array([myround(x) for x in norm_list])
            norm_list[np.where(norm_list > self.df_max)] = self.df_max
            norm_list[np.where(norm_list < self.df_min)] = self.df_min
            norm_list = norm_list.astype(np.int)
        else:
            raise ValueError
        return norm_list

    def __call__(self):
        return self.df


def plot_diff(cols, maptable, model_name=''):
    _order = 'd' if maptable.seg[0] > maptable.seg[1] else 'a'
    rawscore = sorted(maptable.seg)
    raw_label = [str(v) for v in rawscore]
    for f in cols:
        outscore = list(maptable[f+'_ts'])
        outscore = [x if x > -100 else 0 for x in outscore]
        if _order == 'd':
            outscore = outscore[::-1]
        f_rawscore = [x if y > 0 else 0 for x, y in zip(rawscore, outscore)]
        fig, ax = plot.subplots()
        ax.set_title(model_name+'['+f+']: diffrence between raw and out')
        ax.set_xticks(rawscore)
        ax.set_xticklabels(raw_label)
        width = 0.35
        bar_wid = [p - width/2 for p in rawscore]
        bars1 = ax.bar(bar_wid, f_rawscore, width, label=f)
        bar_wid = [p + width/2 for p in rawscore]
        bars2 = ax.bar(bar_wid, outscore, width, label=f+'_ts')
        #
        for i, bars in enumerate([bars1, bars2]):
            for bar in bars:
                if i == 0:
                    notestr = '{}'.format(bar.get_height())
                else:
                    if bar.get_height() > 0:
                        notestr = '{}'.format(int(bar.get_height() - bar.get_x()))
                    else:
                        notestr = '0'
                height = bar.get_height()
                ax.annotate(notestr,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
        ax.legend(loc='upper left', shadow=True, fontsize='x-large')
        fig.tight_layout()
        plot.show()
        return


def plot_score_bar_count(
                         scoretype='raw',
                         cols=None,
                         maptable=None,
                         model_name='',
                         hcolor='r',
                         hwidth=6,
                        ):

    for f in cols:
        if scoretype == 'raw':
            raw_label = [str(x) for x in sorted(maptable.seg)]
            x_data = list(range(max(maptable.seg) + 1))
            df_bin = [maptable.query('seg==' + str(xv))[f + '_count'].values[0]
                      if xv in maptable.seg else 0
                      for xv in x_data]
        else:   # out score
            score_scope = range(min(maptable[f + '_ts']), max(maptable[f+'_ts']+1))
            raw_label = [str(x) for x in score_scope]
            x_data = list(score_scope)
            out_count = maptable.groupby(f+'_ts')[f+'_count'].sum()
            df_bin = [out_count[x] if x in out_count else 0 for x in x_data]

        # create figure and ticks
        fig, ax = plot.subplots()
        ax.set_xticks(x_data)
        ax.set_xticklabels(raw_label)

        # set title
        data = pd.Series()  # create score Series
        for x, y in zip(x_data, df_bin):
            if y > 0:
                data = data.append(pd.Series([x]*int(y)))
        ax.set_title(model_name+' subject:{0}  mean:{1:.2f}, std:{2:.2f}, max:{3:5.1f}, min:{4:5.1f}'.
                     format(f, data.mean(), data.std(), data.max(), data.min())
                     )

        # set count bars
        width = 0.4
        bar_wid = [p + width/2 for p in x_data]
        count_bars = ax.bar(bar_wid, df_bin, width, label=f)

        # set number text above bars
        make_color = 0      # display different color cross bars
        last_height = 0     # adjust height to avoid overlapped number text above bars
        for _bar in count_bars:
            height = _bar.get_height()
            xpos = _bar.get_x() + _bar.get_width() / 2
            note_str = '{}'.format(int(height))
            ypos = 0
            if (height > 100) and abs(height - last_height) < 20:
                if height < last_height:
                    ypos = -10
                else:
                    ypos = +10
            # set count number above bar
            ax.annotate(note_str,
                        xy=(xpos, height),
                        xytext=(0, ypos),              # vertical offset
                        textcoords="offset points",
                        ha='center',
                        va='bottom'
                        )
            # draw red bar inside count-bar
            if make_color == 2:
                yp = height-30 if height > 30 else 0    # avoid to display feint hist height, higher than bar
                plot.plot([xpos, xpos], [0, yp], hcolor, linewidth=hwidth)
                make_color = 0
            else:
                make_color += 1
            last_height = height + ypos
        fig.tight_layout()
        plot.show()
        return


def plot_model(
               col,
               raw_section=(),
               out_section=(),
               down_line=True,
               fontsize=8,
               ):
    """
    显示分段线性转换模型
    """

    plot.rcParams['font.sans-serif'] = ['SimHei']
    plot.rcParams.update({'font.size': fontsize})

    _raw_section = []
    _out_section = []
    for x, y in zip(raw_section, out_section):
        if x[0] > -100:
            _raw_section.append(x)
            _out_section.append(y)

    # calculate formula
    formula_list = []
    for x, y in zip(sorted(_raw_section, key=max), sorted(_out_section, key=max)):
        d = x[1] - x[0]
        if d != 0:
            a = (y[1] - y[0]) / d                   # (y2 - y1) / (x2 - x1)
            b = (y[0] * x[1] - y[1] * x[0]) / d     # (y1x2 - y2x1) / (x2 - x1)
        else:
            a = 0
            b = max(y)                              # mode_section_shrink == 'to_max'
        formula_list.append([(a, b), sorted(x), sorted(y)])

    in_min, in_max = min(min(_raw_section, key=min)), max(max(_raw_section, key=max))
    out_min, out_max = min(min(_out_section, key=max)), max(max(_out_section, key=max))
    # for i, col in enumerate(cols):

    plot.figure(col)
    plot.rcParams.update({'font.size': 10})
    plot.title(u'转换模型({})'.format(col))
    plot.xlim(in_min, in_max)
    plot.ylim(out_min, out_max)
    plot.xlabel(u'\n\n原始分数')
    plot.ylabel(u'转换分数')
    plot.xticks([])
    plot.yticks([])

    # formula = result_dict[col]['coeff']
    # segment map function graph
    for cfi, cf in enumerate(formula_list):
        # line: from x to y
        # cf: (a, b), (x1, x2), (y1, y2)
        x = cf[1]   # if _score_order in ['ascending', 'a'] else cf[1][::-1]
        y = cf[2]   # if _score_order in ['ascending', 'a'] else cf[2][::-1]
        plot.plot(x, y, linewidth=2)

        # line: from endpoint to axis
        for j in [0, 1]:
            # h-line: (x[j], 0) -- (x[j], y[j])
            plot.plot([x[j], x[j]], [0, y[j]], '--', linewidth=1)
            # v-line: (0, y[j]) -- (x[j], y[j])
            plot.plot([0, x[j]], [y[j], y[j]], '--', linewidth=1)

        # label x: raw_score
        for j, xx in enumerate(x):
            # move left if at end point
            plot.text(xx-2 if j == 1 else xx, out_min-2, '{}'.format(int(xx)))
            if x[0] == x[1]:
                break

        # label y: out_score
        for j, yy in enumerate(y):
            plot.text(in_min-2, yy+1 if j == 0 else yy-2, '{}'.format(int(yy)))
            if y[0] == y[1]:
                break

    # darw y = x for showing score shift
    if down_line:
        plot.plot((0, in_max), (0, in_max), 'r--', linewidth=2, markersize=2)

    plot.show()
    return


def plot_dist_seaborn(df, cols, model_name):
    import seaborn as sbn
    for f in cols:
        fig, ax = plot.subplots()
        ax.set_title(model_name+'['+f+']: distribution garph')
        sbn.kdeplot(df[f], shade=True)
        sbn.kdeplot(df[f+'_ts'], shade=True)


class StmPlot:

    def __init__(self, cols=None, maptable=None, raw_section=None, out_setion=None):

        self.plot_name = ['model', 'model', 'raw', 'out', 'diff']

        # self.model_type = None
        # self.outdf = None
        self.model_name = 'sconv'

        self.cols = cols
        self.maptable = maptable
        self.raw_section = raw_section
        self.out_section = out_setion

    def plot(self, mode='model'):
        if mode not in self.plot_name:
            plot.figure('invalid mode {}!'.format(mode))
            return
        if mode == 'model':
            for col in self.cols:
                plot_model(col=col,
                           raw_section=self.raw_section,
                           out_section=self.out_section,
                           down_line=True)
        elif mode == 'raw':
            plot_score_bar_count(
                scoretype='raw',
                cols=self.cols,
                maptable=self.maptable,
                model_name=self.model_name
            )
        elif mode == 'out':
            plot_score_bar_count(
                scoretype='out',
                cols=self.cols,
                maptable=self.maptable,
                model_name=self.model_name
            )
        elif mode == 'diff':
            plot_diff(
                model_name=self.model_name,
                cols=self.cols,
                maptable=self.maptable
            )


class Formula:
    # used to caculate transformed score

    # Constant Number
    Tiny_Value = 10 ** -8
    Invalid_Score = -1000

    @classmethod
    def formula1(cls, raw, x, y, decimals):
        """
        formula-1
        y = a*x + b
        a = (y2-y1)/(x2-x1)
        b = -x1/(x2-x1) + y1
        """
        denom = x[1]-x[0]
        if abs(denom) > cls.Tiny_Value:
            return round45((y[1]-y[0])/denom * raw - x[0]/denom + y[0], decimals)
        else:
            return cls.Invalid_Score

    @classmethod
    def formula2(cls, raw, x, y, decimals):
        """
        original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
        formula2: y = a*(x - b) + c
                  a = (y2-y1)/(x2-x1)
                  b = x1
                  c = y1
        """

        denom = x[1]-x[0]
        if abs(denom) > cls.Tiny_Value:
            return round45((y[1] - y[0]) / denom * (raw - x[0]) + y[0], decimals)
        return cls.Invalid_Score

    @classmethod
    def formula3(cls, raw, x, y, decimals):
        """
        recommend to use,  int/int to float
        original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
        formula3: y = (a*x + b) / c
                  a=(y2-y1)
                  b=y1x2-y2x1
                  c=(x2-x1)
        """
        denom = x[1] - x[0]
        if abs(denom) > cls.Tiny_Value:
            return round45(((y[1]-y[0]) * raw + y[0]*x[1] - y[1]*x[0]) / denom, decimals)
        return cls.Invalid_Score


def models_hist(font_size=12):
    from sconv import models as m

    def __model_describe(name='shandong'):
        __ratio = m.Models[name].ratio
        __section = m.Models[name].section
        if name == 'b900':
            __mean, __std, __skewness = 500, 100, 0
        elif name == 'b300':
            __mean, __std, __skewness = 180, 30, 0
        else:
            samples = []
            [samples.extend([np.mean(s)] * int(__ratio[_i])) for _i, s in enumerate(__section)]
            __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))
        return __mean, __std, __skewness

    _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'p7', 'b900']

    ms_dict = dict()
    for _name in _names:
        ms_dict.update({_name: __model_describe(name=_name)})

    plot.figure('New Gaokao Score Models: name(mean, std, skewness)')
    plot.rcParams.update({'font.size': font_size})
    for i, k in enumerate(_names):
        plot.subplot(240+i+1)
        _wid = 2
        if k in ['shanghai']:
            x_data = range(40, 71, 3)
        elif k in ['zhejiang', 'beijing', 'tianjin']:
            x_data = range(40, 101, 3)
        elif k in ['shandong']:
            x_data = [x for x in range(26, 100, 10)]
            _wid = 8
        elif k in ['guangdong']:
            x_data = [np.mean(x) for x in m.Models[k].section][::-1]
            _wid = 10
        elif k in ['p7']:
            x_data = [np.mean(x) for x in m.Models[k].section][::-1]
            _wid = 10
        elif k in ['b900']:
            x_data = [x for x in range(100, 901)]
            _wid = 1
        elif k in ['b300']:
            x_data = [x for x in range(60, 301)]
            _wid = 1
        else:
            raise ValueError(k)
        plot.bar(x_data, m.Models[k].ratio[::-1], width=_wid)
        plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))


def make_maptable_doc(maptable, col_width=10, seg_decimal=0, pecent_decimal=6, ts_decimal=0, sep=False):
    """
    保存分数转换映射表为文档文件
    save map talbe to text doc file
    # deprecated: use module ptt to create griding and  paging text doc
    # with open(filename, mode='w', encoding='utf-8') as f:
    #     f.write(ptt.make_mpage(self.maptable, page_line_num=50))
    """

    columns_list = list(maptable.columns)
    t = ' '
    for cname in maptable.columns:
        _cname = cname if len(cname) <= col_width else cname[:col_width]
        if cname == 'seg':
            _w = 5
        elif ('_coun' in cname) or ('_sum' in cname):
            _w = 10
        elif '_ts' in cname:
            _w = 10
        else:
            _w = col_width
        t += ('{}'.format(_cname)).center(_w)
    t += '\n'

    start = False
    wtext = ''
    for row_no, row in maptable.iterrows():
        s = '|'
        for ci, col in enumerate(row):
            if columns_list[ci] == 'seg':
                _fstr = '{:5.' + str(seg_decimal) + 'f}'
                s += _fstr.format(col)
            elif ('_count' in columns_list[ci]) or ('_sum' in columns_list[ci]):
                s += ('{:5d}'.format(int(col))).center(10)
            elif '_ts' in columns_list[ci]:
                _fstr = '{:8.' + str(ts_decimal) + 'f}  '
                s += _fstr.format(col)
            elif isinstance(col, float):
                s += ('{:.8f}'.format(col)).rjust(col_width)
            else:
                s += ('{}'.format(col)).rjust(col_width)
        s += '|'
        if not start:
            wtext += t
            wtext += '-' * len(s) + '\n'
            start = True
        wtext += s + '\n'
        if sep:
            wtext += '-' * len(s) + '\n'
    if not sep:
        wtext += s + '\n' + '-' * len(s) + '\n'

    return wtext
