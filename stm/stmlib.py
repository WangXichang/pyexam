# coding: utf-8


import numpy as np
import pandas as pd
import time
import os
import re
import numbers
import configparser as cfp
from collections import namedtuple
import scipy.stats as sts
import logging
from logging import handlers
import importlib as pb


# call SegTable.run() return instance of SegTable
def run_seg(
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
        seglist = [x for x in range(int(self.__segMin), int(self.__segMax + 1))]
        if self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__outdfframe = pd.DataFrame({'seg': seglist})
        outdf = self.__outdfframe
        for f in self.__cols:
            # calculate preliminary group count
            tempdf = self.df
            tempdf.loc[:, f] = tempdf[f].apply(round45r)

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
                outdf[f + '_sum'].apply(lambda x: round45r(x/maxsum, self.__percent_decimal))
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


def round45r(number, digits=0):
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


def set_ellipsis_in_digits_sequence(digit_seq):
    _digit_seq = None
    if type(digit_seq) == str:
        _digit_seq = tuple(int(x) for x in digit_seq)
    elif type(digit_seq) in (list, tuple):
        _digit_seq = digit_seq
    else:
        print('digit_seq error type: {}'.format(type(digit_seq)))
        raise ValueError
    ellipsis_list = []
    if len(_digit_seq) > 0:
        start_p, end_p, count_p = -1, -1, -1
        for p in _digit_seq:
            if p == _digit_seq[0]:
                start_p, end_p, count_p = p, p, 1
            if p == _digit_seq[-1]:
                if count_p == 1:
                    ellipsis_list += [start_p, p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p, p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p, p]
                elif p == end_p + 1:
                    ellipsis_list += [start_p, Ellipsis, p]
                else:
                    ellipsis_list += [start_p, Ellipsis, end_p, p]
                break
            if p > end_p + 1:
                if count_p == 1:
                    ellipsis_list += [start_p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p]
                elif count_p == 3:
                    ellipsis_list += [start_p, end_p-1, end_p]
                else:
                    ellipsis_list += [start_p, Ellipsis, end_p]
                count_p = 1
                start_p, end_p = p, p
            elif p == end_p + 1:
                end_p, count_p = p, count_p + 1
    return str(ellipsis_list).replace('Ellipsis', '...')


def get_norm_section_pdf(
                    start=21,
                    end=100,
                    section_num=8,
                    std_num=2.6,
                    add_cutoff=True,
                    model_type='plt',
                    ratio_coeff=1,      # 1, or 100
                    sort_order='d',
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
       [0.03000, 0.07513, 0.16036, 0.234265, ..., 0.03000],
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
    :return: namedtuple('result', ('section':((),...), 'pdf': (), 'cdf': (), 'cutoff': float, 'add_cutoff': bool))
    """
    _mean, _std = (end + start) / 2, (end - start) / 2 / std_num
    section_point_list = np.linspace(start, end, section_num + 1)
    cutoff = sts.norm.cdf((start - _mean) / _std)
    pdf_table = [0]
    cdf_table = [0]
    last_pos = (start - _mean) / _std
    _cdf = 0
    for i, pos in enumerate(section_point_list[1:]):
        _zvalue = (pos - _mean) / _std
        this_section_pdf = sts.norm.cdf(_zvalue) - sts.norm.cdf(last_pos)
        if (i == 0) and add_cutoff:
            this_section_pdf += cutoff
        pdf_table.append(this_section_pdf)
        cdf_table.append(this_section_pdf + _cdf)
        last_pos = _zvalue
        _cdf += this_section_pdf
    if add_cutoff:
        pdf_table[0] += cutoff
        cdf_table[-1] = 1
    if model_type == 'plt':
        section_list = [(x, y) if i == 0 else (x + 1, y)
                        for i, (x, y) in enumerate(zip(section_point_list[:-1], section_point_list[1:]))]
    else:
        section_list = [(x, x) for x in section_point_list]
    if ratio_coeff != 1:
        pdf_table = [x * ratio_coeff for x in pdf_table]
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


def isfilestr(fstr):
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

def read_conf(conf_name):
    mcfg = dict()
    cfper = dict()

    if os.path.isfile(conf_name):
        cfper = cfp.ConfigParser()
        cfper.read(conf_name)

    if 'model_in' in cfper.keys():
        if 'name' in cfper['model_in']:
            mcfg.update({'model_name': remove_annotation(cfper['model_in']['name'])})
            mcfg.update({'model_in_check': True})
        else:
            mcfg.update({'model_in_check': False})

    if not mcfg['model_in_check']:
        model_list = ['name', 'type', 'ratio', 'section', 'desc']
        if 'model_new' in cfper.keys():
            new_set = True
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
                else:
                    new_set = False
                    print('config error: model para incomplete in model_new: {}'.format(k))
            if new_set:
                _ch = Checker.check_model_para(
                    model_type=mcfg['model_new_type'],
                    model_ratio=mcfg['model_new_ratio'],
                    model_section=mcfg['model_new_section'],
                    model_desc=mcfg['model_new_desc']
                    )
                mcfg.update({'model_new_check': _ch})

    if 'data' in cfper.keys():
        df = None
        if 'df' in cfper['data']:
            dffile = remove_annotation(cfper['data']['df'])
            mcfg.update({'dffile': dffile})
            if os.path.isfile(dffile):
                try:
                    df = pd.read_csv(dffile)
                    mcfg.update({'df': df})
                except:
                    print('data read error!')
            else:
                print('invalid file name: {}'.format(dffile))
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

    if 'para' in cfper.keys():
        for _para in cfper['para']:
            mcfg.update({_para: remove_annotation(cfper['para'][_para])})

        if 'raw_score_min' in mcfg.keys():
            mcfg['raw_score_min'] = int(mcfg['raw_score_min'])
        else:
            mcfg.update({'raw_score_min': 0})

        if 'raw_score_max' in mcfg.keys():
            mcfg['raw_score_max'] = int(mcfg['raw_score_max'])
        else:
            mcfg.update({'raw_score_max': 100})

        if 'out_score_decimals' in mcfg.keys():
            mcfg['out_score_decimals'] = int(mcfg['out_score_decimals'])
        else:
            mcfg.update({'out_score_decimals': 0})

        if 'tiny_value' in mcfg.keys():
            mcfg['tiny_value'] = eval(mcfg['tiny_value'])
        else:
            mcfg['tiny_value'] = 10**-10

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

        # set bool
        bool_list = ['logdisp', 'logfile', 'verify', 'saveresult']
        default_d = {'logdisp': True,
                     'logfile': False,
                     'verify': False,
                     'saveresult': True,
                     }
        for ks in bool_list:
            if ks in mcfg.keys():
                if mcfg[ks].lower() in ['false', '0', '']:
                    mcfg[ks] = False
                else:
                    mcfg[ks] = True
            else:
                mcfg.update({ks: default_d[ks]})

    if 'strategy' in cfper.keys():
        for _mode in cfper['strategy']:
            _mode_str = remove_annotation(cfper['strategy'][_mode])
            mcfg.update({_mode: _mode_str})

    return mcfg


def remove_annotation(s):
    p = s.find('#')
    rs = s.lower()
    if p >= 0:
        rs = s[0:p].strip().lower()
    return rs


def make_config_file(filename):
    template = \
        """
        [task]
        logname =                           # used to add in logfile name: [model_name]_[logname]_[year]_[month]_[day].log
        logdisp = 1                         # output message to consol or not
        logfile = 1                         # output message to log file or not
        verify = 0                          # use dual algorithm to verify result or not
        savedf = True
        savereport = True
        savemaptable = True

        
        [model_in]
        namex = shandong                    # model name biult in models
        
        
        [data]
        df = df             # file name, used to read to DataFrame
        cols = km1, km2     # score fields to transform score
        
        
        [para]
        raw_score_min = 0                   # min score for raw score
        raw_score_max = 100                 # max score for raw score
        out_score_decimals = 0              # decimal digits for out score
        tiny_value = 10 ** -10              # smallest value for precision in calculation process
        
        
        [strategy]
        mode_ratio_prox = upper_min         # ('upper_min', 'lower_max', 'near_max', 'near_min')
        mode_ratio_cumu = no                # ('yes', 'no')
        mode_sort_order = d                 # ('d', 'a')
        mode_section_point_first = real     # ('real', 'defined')
        mode_section_point_start = step     # ('step', 'share')
        mode_section_point_last = real      # ('real', 'defined')
        mode_section_degraded = to_max      # ('to_max', 'to_min', 'to_mean')
        mode_section_lost = real            # ('real', 'zip')
        
        
        [model_new]
        name = test-similar-shandong        # model name
        type = plt                          # model type, valid value: plt, ppt, pgt
        # section for out score, point-pair tuple separated by comma
        section = (120, 111), (110, 101), (100, 91), (90, 81), (80, 71), (70, 61), (60, 51), (50, 41)
        # ratio for each section, sum=100
        ratio = 3, 7, 16, 24, 24, 16, 9, 1
        # description for model
        desc = new model for test, similar to Shandong
        """

    if isfilestr(filename):
        with open(filename, 'a') as fp:
            top_line = True
            for ss in template.split('\n'):
                if len(ss.strip()) == 0:
                    continue
                if ('[' in ss) and not top_line:
                    fp.write('\n'*3 + ss.strip() + '\n')
                else:
                    fp.write(ss.strip() + '\n')
                top_line = False


class Checker:

    @staticmethod
    def check_run(
            model_name='shandong',
            df=None,
            cols=None,
            mode_ratio_prox='upper_min',
            mode_ratio_cumu='no',
            mode_sort_order='d',
            mode_section_point_first='real',
            mode_section_point_start='step',
            mode_section_point_last='real',
            mode_section_degraded='map_to_max',
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
        if model_name.lower() not in models.Models.keys():
            logger.loginfo('error name: name={} not in models.Models!'.format(model_name))
            return False

        # check input data: DataFrame
        if not Checker.check_df_cols(df, cols, raw_score_range, logger):
            return False

        # check strategy
        if not Checker.check_strategy(
                mode_ratio_prox=mode_ratio_prox,
                mode_ratio_cumu=mode_ratio_cumu,
                mode_sort_order=mode_sort_order,
                mode_section_point_first=mode_section_point_first,
                mode_section_point_start=mode_section_point_start,
                mode_section_point_last=mode_section_point_last,
                mode_section_degraded=mode_section_degraded,
                mode_section_lost=mode_section_lost,
                logger=logger,
                models=models,
        ):
            return False

        if out_score_decimal_digits < 0 or out_score_decimal_digits > 10:
            logger.logger.info('warning: decimal digits={} set may error!'.format(out_score_decimal_digits))

        return True

    @staticmethod
    def reload_stm_modules(logger=None, modules=None):
        try:
            for m in modules:
                # print('reload:{}'.format(m))
                pb.reload(m)
        except:
            return False
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
        if model_type not in ['ppt', 'plt', 'pgt']:
            logger.loginfo('error type: valid type must be in {}'.format(model_type, ['ppt', 'plt', 'pgt']))
            return False

        # check ratio
        if model_type == 'pgt':
            if len(model_ratio) == 0:
                logger.loginfo('error ratio: length == 0 in model={}!'.format(model_type))
                return False
            if model_ratio[0] < 0 or model_ratio[0] > 100:
                logger.loginfo('error ratio: in type=tai, ratrio[0]={} must be range(0, 101) as the percent of top score ratio!'.format(model_ratio[0]))
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
            mode_sort_order='descending',
            mode_section_point_first='real',
            mode_section_point_start='step',
            mode_section_point_last='real',
            mode_section_degraded='map_to_max',
            mode_section_lost='ignore',
            logger=None,
            models=None,
            ):

        if logger is None:
            logger = get_logger('check')
            logger.logging_consol = True
            logger.logging_file = False

        st = {'mode_ratio_prox': mode_ratio_prox,
              'mode_ratio_cumu':mode_ratio_cumu,
              'mode_sort_order': mode_sort_order,
              'mode_section_point_first': mode_section_point_first,
              'mode_section_point_start': mode_section_point_start,
              'mode_section_point_last': mode_section_point_last,
              'mode_section_degraded': mode_section_degraded,
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
                logger.loginfo('error data: df is not pandas.DataFrame!')
                return False
        if len(df) == 0:
            logger.loginfo('error data: df is empty!')
            return False
        if type(cols) not in (list, tuple):
            logger.loginfo('error type: cols must be list or tuple, real type is {}!'.format(type(cols)))
            return False
        for col in cols:
            if type(col) is not str:
                logger.loginfo('error col: {} is not str!'.format(col), 'error')
                return False
            else:
                if col not in df.columns:
                    logger.loginfo('error col: {} is not in df.columns!'.format(col), 'error')
                    return False
                if not isinstance(df[col][0], numbers.Real):
                    logger.loginfo('type error: column[{}] not Number type!'.format(col), 'error')
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
        task_str = 'log_'
    else:
        task_str = logname + '_'
    log_file = \
        task_str + model_name + '_' + \
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
                 filename='log_test.log',
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

    def loginfo(self, ms='', level='info'):
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
        self.set_handlers(self.logger_format)

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
