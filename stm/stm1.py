# -*- utf-8 -*-


# comments to module
"""
    2018.09.24 -- 2018.11
    2019.09.03 --
    designed for new High Test grade score model
    also for shandong interval linear transform

    stm模块说明：

    [classes] 模块中的类
       PltScore: 分段线性转换模型, 新高考改革使用
       TaiScore: 台湾等级分数模型 Taiwan college entrance test and middle school achievement test model
         Zscore: Z分数转换模型 zscore model
         Tscore: T分数转换模型 t_score model
"""


# built-in import
import copy
import time
import os
import fractions as fr
import bisect as bst
import array
import abc


# external import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
import seaborn as sbn

# stm import
from stm import stmlib as slib

import warnings
warnings.filterwarnings('ignore')


# Score Transform Model Interface
# Abstract class
class ScoreTransformModel(abc.ABC):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    基于该类的子类（转换分数模型）：
        Z分数非线性模型(Zscore)
        T分数非线性模型(Tscore）
        新高考分段线性等级分数转换分数模型（PltScore）
        param model_name: type==str, in model_setin.Models.keys or models_ext.Models.keys
        param model_type: type==str, in ['plt', 'ppt', 'pgt']
        param df:         type==DataFrame, raw score data,
        param cols:       type==list, fields in df, assign somr subjects score to transform
        param outdf:      type==DataFrame, transform score data,
    """

    def __init__(self, model_name='', model_type=''):
        self.model_name = model_name
        self.model_type = model_type

        self.df = pd.DataFrame()
        self.cols = []
        self.map_table = pd.DataFrame()
        self.outdf = pd.DataFrame()

        self.raw_score_defined_min = None
        self.raw_score_defined_max = None

        self.out_decimal_digits = 0
        self.out_report_doc = ''

    @abc.abstractmethod
    def set_data(self, df=None, cols=None):
        """
        设置输入数据框架(df: dataframe)和进行转换的数据列（cols: list）
        """

    @abc.abstractmethod
    def set_para(self, *args, **kwargs):
        """
        设置转换分数的参数
        set parameters used to transform score
        """

    @abc.abstractmethod
    def report(self):
        """
        返回系统生成的分数转换报告
        return score transforming report created by model running
        """

    def run(self):
        """
        run to get map_table, formula, outdf...
        """

    def read_df_from_csv(self, filename=''):
        if not os.path.isfile(filename):
            print('filename:{} is not a valid file name or the file not exists!'.format(filename))
            return
        self.df = pd.read_csv(filename)

    def save_outdf_to_csv(self, filename):
        self.outdf.to_csv(filename, index=False)

    def save_report_doc(self, filename):
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(self.out_report_doc)

    def save_map_table_doc(self, filename, col_width=20):
        """
        保存分数转换映射表为文档文件
        save map talbe to text doc file
        # deprecated: use module ptt to create griding and  paging text doc
        # with open(filename, mode='w', encoding='utf-8') as f:
        #     f.write(ptt.make_mpage(self.map_table, page_line_num=50))
        """
        with open(filename, mode='w', encoding='utf-8') as f:
            t = ' '
            for cname in self.map_table.columns:
                t += ('{}'.format(cname)).rjust(col_width)
            t += '\n'
            start = False
            for row_no, row in self.map_table.iterrows():
                s = '|'
                for col in row:
                    if isinstance(col, float):
                        s += ('{:.8f}'.format(col)).rjust(col_width)
                    else:
                        s += ('{}'.format(col)).rjust(col_width)
                s += '|'
                if not start:
                    f.write(t)
                    f.write('-'*len(s) + '\n')
                    start = True
                f.write(s + '\n' + '-'*len(s) + '\n')

    def plot(self, mode='raw'):
        # implemented plot_out, plot_raw score figure
        if mode.lower() == 'out':
            self.plot_out_score()
        elif mode.lower() == 'raw':
            self.plot_raw_score()
        # return False so that implementing other plotting in subclass
        else:
            return False
        # do not need to implement in subclass
        return True

    def plot_out_score(self):
        if not self.cols:
            print('no field assigned in {}!'.format(self.df))
            return
        fig_title = 'Out Score '
        for fs in self.cols:
            plot.figure(fs)
            if fs + '_ts' in self.outdf.columns:  # find sf_out_score field
                sbn.distplot(self.outdf[fs + '_ts'])
                plot.title(fig_title + fs)
            elif fs + '_grade' in self.outdf.columns:  # find sf_out_score field
                sbn.distplot(self.outdf[fs + '_grade'])
                plot.title(fig_title + fs)
            else:
                print('no out score fields found in outdf columns:{}!'.
                      format(list(self.outdf.columns)))
        return

    def plot_raw_score(self):
        if not self.cols:
            print('no field assign in rawdf!')
            return
        labelstr = 'Raw Score '
        for sf in self.cols:
            plot.figure(sf)
            sbn.distplot(self.df[sf])
            plot.title(labelstr + sf)
        return


# piecewise linear transform model
    # ShanDong Model Analysis
    # ratio = [3, 7, 16, 24, 24, 16, 7, 3], grade = [(21, 30),(31, 40), ..., (91, 100)]
    # following is estimation to std:
        # according to percent
        #   test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #   test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #   test std=15.950713502      at 30    Zcdf(-30/std)=0.03
        # according to std
        #   cdf(100)= 0.99496 as std=15.54375, 0.9939 as std=15.9507
        #   cdf(90) = 0.970(9)79656 as std=15.9507135,  0.972718 as std=15.606,  0.9731988 as std=15.54375
        #   cdf(80) = 0.900001195 as std=15.606,  0.9008989 as std=15.54375
        #   cdf(70) = 0.0.73999999697 as std=15.54375
        #   cdf(60) = 0.0
        #   cdf(50) = 0.26  +3.027*E-9 as std=15.54375
        #   cdf(40) = 0.0991 as std=15.54375
        #   cdf(30) = 0.0268 as std=15.54375
        #   cdf(20) = 0.0050 as std=15.54375
        # some problems:
        #   p1: std scope in 15.5-16
        #   p2: cut percent at 20, 100 is a little big, so std is reduced
        #   p3: percent at 30,40 is a bit larger than normal according to std=15.54375
        # on the whole, fitting is approximate fine
        # set model score percentages and endpoints
        # get approximate normal distribution
        # according to percent , test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #                        test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #                        test std=15.950713502      at 30    Zcdf(-30/std)=0.03
        # according to std
        #   cdf(100)= 0.99496           as std=15.54375,    0.9948      as std=15.606       0.9939    as std=15.9507
        #   cdf(90) = 0.970(9)79656     as std=15.9507135   0.97000     as std=15.54375     0.972718    as std=15.606
        #   cdf(80) = 0.900001195       as std=15.606,      0.9008989   as std=15.54375
        #   cdf(70) = 0.0.73999999697   as std=15.54375
        #   cdf(60) = 0.0
        #   cdf(50) = 0.26+3.027*E-9    as std=15.54375
        #   cdf(40) = 0.0991            as std=15.54375
        #   cdf(30) = 0.0268            as std=15.54375
        #   cdf(20) = 0.0050            as std=15.54375
        # ---------------------------------------------------------------------------------------------------------
        #     percent       0      0.03       0.10      0.26      0.50    0.74       0.90      0.97       1.00
        #   std/points      20      30         40        50        60      70         80        90         100
        #   15.54375    0.0050   0.0268       0.0991   [0.26000]   0    0.739(6)  0.9008989  0.97000    0.99496
        #   15.6060     0.0052   0.0273      [0.09999]  0.26083    0    0.73917   0.9000012  0.97272    0.99481
        #   15.9507     0.0061  [0.0299(5)]   0.10495   0.26535    0    0.73465   0.8950418  0.970(4)   0.99392
        # ---------------------------------------------------------------------------------------------------------
        # on the whole, fitting is approximate fine
        # p1: std scope in 15.54 - 15.95
        # p2: cut percent at 20, 100 is a little big, std would be reduced
        # p3: percent at 30 is a bit larger than normal according to std=15.54375, same at 40
        # p4: max frequency at 60 estimation:
        #     percentage in 50-60: pg60 = [norm.pdf(0)=0.398942]/[add:pdf(50-60)=4.091] = 0.097517
        #     percentage in all  : pga = pg60*0.24 = 0.023404
        #     peak frequency estimation: 0.0234 * total_number
        #          max number at mean nerghbor point:200,000-->4680,   300,000 --> 7020

    # GuangDong Model Analysis
    # ratio = [2, 15, 33, 33, 17]
    # gradescore = [(30, 40), (41, 58), (59, 70), (71, 82), (83, 100)]
    # meanscore ~= 35*0.02+48.5*0.13+64.5*0.35+76.5*0.35+92.5*0.15 = 70.23

    # SS7 Model Analysis
    # ratio = [2, 13, 35, 35, 15]
    # gradescore = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]
    # meanscore ~= 35*0.02+48*0.13+63*0.35+78*0.35+93*0.15 = 70.24

class PltScore(ScoreTransformModel):
    """
    PltModel:
    linear transform from raw-score to grade-score by each intervals
    intervals are created according to ratio values
    calculation procedure:
    (1) set input data dataframe by set_data
    (2) set parameters by set_para
    (3) use run() to calculate
    (4) result data: outdf, with raw data fields and output fields [raw_score_field]_ts
                     result_dict, {[score_field]:{'coeff':[(a, b), (x1, x2), (y1, y2)],
                                                  'formula': 'a*(x - x1) + b',
                                                  }
    """

    def __init__(self, model_name='', model_type=''):
        # intit df, outdf, model_name, model_type
        super(PltScore, self).__init__(model_name, model_type)

        # new properties
        self.raw_score_ratio_cum = []
        self.out_score_section = []
        self.out_decimal_digits = 0
        self.out_score_max = None
        self.out_score_min = None
        self.raw_score_real_max = None
        self.raw_score_real_min = None
        self.raw_score_defined_max = None
        self.raw_score_defined_min = None

        # strategy
        self.strategy_dict = dict()

        # run control
        self.display=True
        self.logger=None

        # calc precision
        self.tiny_value = 10**-8

        # result
        self.map_table = pd.DataFrame()
        self.result_raw_endpoints = []
        self.result_ratio_dict = dict()
        self.result_formula_coeff = dict()
        self.result_formula_text_list = ''
        self.result_dict = dict()

    # plt
    def set_data(self, df=None, cols=None):

        # check and set rawdf
        if type(df) == pd.Series:
            self.df = pd.DataFrame(df)
        elif type(df) == pd.DataFrame:
            self.df = df
        else:
            if self.logger:
                self.logger.loginfo('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set outdf
        if type(cols) is str:
            if cols in self.df.columns:
                self.cols = [cols]
                return True
            else:
                if self.logger:
                    self.logger.loginfo('invalid field name in cols: {}'.format(cols))
        elif type(cols) not in [list, tuple]:
            self.loginfo('col set fail!\n not a list or tuple!')
            return False
        elif sum([1 if sf in df else 0 for sf in cols]) != len(cols):
            self.loginfo('field of cols not in rawdf.columns!')
            return False
        else:
            self.cols = cols
            return True

    # plt
    def set_para(self,
                 raw_score_ratio=None,
                 out_score_section=None,
                 raw_score_defined_min=0,
                 raw_score_defined_max=100,
                 mode_ratio_prox='upper_min',
                 mode_ratio_cumu='no',
                 mode_sort_order='descending',
                 mode_section_point_first='real',
                 mode_section_point_start='step',
                 mode_section_point_last='real',
                 mode_section_degraded='to_max',
                 mode_section_lost='real',
                 mode_score_zero='real',
                 # mode_seg_end_share='no',
                 out_decimal_digits=None,
                 display=True,
                 tiny_value=10**-8,
                 logger=None
                 ):
        if isinstance(display, bool):
            self.display=display

        self.out_decimal_digits = out_decimal_digits

        if mode_sort_order in ['descending', 'd']:
            raw_p = raw_score_ratio
            self.out_score_section = out_score_section
        else:
            raw_p = raw_score_ratio[::-1]
            self.out_score_section = tuple(x[::-1] for x in out_score_section[::-1])
        ratio_sum = sum(raw_p)
        self.raw_score_ratio_cum = [sum(raw_p[0:x + 1])/ratio_sum for x in range(len(raw_p))]

        self.raw_score_defined_min, self.raw_score_defined_max = raw_score_defined_min, raw_score_defined_max

        self.strategy_dict['mode_ratio_prox'] = mode_ratio_prox
        self.strategy_dict['mode_ratio_cumu'] = mode_ratio_cumu
        self.strategy_dict['mode_sort_order'] = mode_sort_order
        self.strategy_dict['mode_section_point_first'] = mode_section_point_first
        self.strategy_dict['mode_section_point_start'] = mode_section_point_start
        self.strategy_dict['mode_section_point_last'] = mode_section_point_last
        self.strategy_dict['mode_section_degraded'] = mode_section_degraded
        self.strategy_dict['mode_section_lost'] = mode_section_lost
        # self.strategy_dict['mode_score_zero'] = mode_score_zero

        self.tiny_value = tiny_value
        self.logger = logger

    def loginfo(self, ms):
        if self.logger:
            self.logger.loginfo(ms)
        else:
            if self.display:
                print(ms)

    # --------------data and para setting end

    # plt score run
    def run(self):

        self.loginfo('stm start ...\n'+'-'*120)
        stime = time.time()

        if self.out_score_section is not None:
            self.out_score_real_max = max([max(x) for x in self.out_score_section])
            self.out_score_real_min = min([min(x) for x in self.out_score_section])

        # calculate seg table
        # self.loginfo('calculating map_table ...')
        _segsort = 'a' if self.strategy_dict['mode_sort_order'] in ['ascending', 'a'] else 'd'
        seg_model = slib.run_seg(
                  df=self.df,
                  cols=self.cols,
                  segmax=self.raw_score_defined_max,
                  segmin=self.raw_score_defined_min,
                  segsort=_segsort,
                  segstep=1,
                  display=False,
                  usealldata=False
                  )
        self.map_table = seg_model.outdf   # .copy(deep=True)

        # create field_fr in map_table
        #   strange error!!: some seg percent to zero
        #   self.map_table[f+'_percent'] = self.map_table[f+'_fr'].apply(lambda x: float(x))
        # for f in self.cols:
        #     max_sum = max(self.map_table[f+'_sum'])
        #     max_sum = 1 if max_sum == 0 else max_sum
        #     self.map_table[f+'_fr'] = \
        #         self.map_table[f+'_sum'].apply(lambda x: fr.Fraction(x, max_sum))
            # self.map_table.astype({f+'_fr': fr.Fraction})     # encounter error in python 3.7.4

        # transform score on each field
        self.result_dict = dict()
        self.outdf = self.df.copy(deep=True)
        for i, col in enumerate(self.cols):
            self.loginfo('transform score: {0} => {0}_ts'.format(col))

            # there is a problem: max set to locale value to each col
            self.raw_score_real_max = self.df[col].max()
            self.raw_score_real_min = self.df[col].min()

            # get formula and save
            _get_formula = False
            if self.model_type == 'ppt': # msin.Models[self.model_name].type == 'ppt':
                # self.loginfo('get ppt formula ...')
                _get_formula = self.get_formula_ppt(col)
            elif self.model_type == 'plt':
                # self.loginfo('get plt formula ...')
                _get_formula = self.get_formula_plt(col)
            else:
                self.loginfo('error model type: not supported type={}'.format(self.model_type))
                return None
            if not _get_formula:
                self.loginfo('error calc: getting formula fail !')
                return None

            # get field_ts in outdf
            self.outdf.loc[:, (col + '_ts')] = \
                self.outdf[col].apply(
                    lambda x: self.get_ts_score_from_formula(col, x))

        # create col_ts in map_table
        df_map = self.map_table
        for col in self.cols:
            # self.loginfo('   calculate: map_table[{0}] => [{0}_ts]'.format(col))
            col_name = col + '_ts'
            df_map.loc[:, col_name] = df_map['seg'].apply(
                lambda x: self.get_ts_score_from_formula(col, x))

        # make report doc
        self.make_report()

        self.loginfo('stm end, elapsed-time:{:.4f}'.format(time.time() - stime) + '\n' + '-'*120)
        self.loginfo('Report\n' + '-'*120)
        self.loginfo(self.out_report_doc)

    # run end

    # -----------------------------------------------------------------------------------
    # formula-1
    # y = a*x + b
    # a = (y2-y1)/(x2-x1)
    # b = -x1/(x2-x1) + y1
    def get_ts_score_from_formula_ax_b(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                return slib.round45r(cf[0][0] * x + cf[0][1])
        return -1

    # -----------------------------------------------------------------------------------
    # formula-2
    # y = a*(x - b) + c
    # a = (y2-y1)/(x2-x1)
    # b = x1
    # c = y1
    def get_ts_score_from_formula_ax_b_c(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                v = (cf[1][1]-cf[1][0])
                if v == 0:
                    return slib.round45r(cf[0][1])
                a = (cf[2][1]-cf[2][0])/v
                b = cf[1][0]
                c = cf[2][0]
                return slib.round45r(a * (x - b) + c)
        return -1

    # -----------------------------------------------------------------------------------
    # formula-3 new, recommend to use,  int/int to float
    # original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
    # variant:  y = (a*x + b) / c
    #           a=(y2-y1)
    #           b=y1x2-y2x1
    #           c=(x2-x1)
    def get_ts_score_from_formula(self, field, x):
        if x > self.raw_score_defined_max:
            # raise ValueError
            return self.out_score_real_max
        if x < self.raw_score_defined_min:
            # raise ValueError
            return self.out_score_real_min
        for cf in self.result_dict[field]['coeff'].values():
            if (cf[1][0] <= x <= cf[1][1]) or (cf[1][0] >= x >= cf[1][1]):
                a = (cf[2][1]-cf[2][0])
                b = cf[2][0]*cf[1][1] - cf[2][1]*cf[1][0]
                c = (cf[1][1]-cf[1][0])
                # x1 == x2: use mode_section_degraded: max, min, mean(y1, y2)
                if c == 0:
                    if self.strategy_dict['mode_section_degraded'] == 'to_max':
                        return slib.round45r(max(cf[2]), self.out_decimal_digits)
                    elif self.strategy_dict['mode_section_degraded'] == 'to_min':
                        return slib.round45r(min(cf[2]), self.out_decimal_digits)
                    elif self.strategy_dict['mode_section_degraded'] == 'to_mean':
                        return slib.round45r(np.mean(cf[2]))
                    else:
                        # invalid mode
                        return None
                return slib.round45r((a * x + b) / c, self.out_decimal_digits)
        # raw score not in coeff[1]
        return -1000

    # formula hainan, each segment is a single point
    # y = x for x in [x, x]
    # coeff: (a=0, b=x), (x, x), (y, y))
    # len(ratio_list) = len(map_table['seg'])
    def get_formula_ppt(self, col):
        self.result_raw_endpoints = [x for x in self.map_table['seg']]
        self.map_table.loc[:, col+'_ts'] = -1
        coeff_dict = dict()
        result_ratio = []
        tiny_value = self.tiny_value     # used to judge zero(s==0) or equality(s1==s2)

        _mode_sort = self.strategy_dict['mode_sort_order']

        if _mode_sort in ['d', 'descending']:
            _mode_ppt_score_min = self.strategy_dict['mode_section_point_last']  # real_min or paper_min
            _mode_ppt_score_max = self.strategy_dict['mode_section_point_first']  # real_max or paper_max
        else:
            _mode_ppt_score_max = self.strategy_dict['mode_section_point_last']  # real_min or paper_min
            _mode_ppt_score_min = self.strategy_dict['mode_section_point_first']  # real_max or paper_max

        _mode_prox = self.strategy_dict['mode_ratio_prox']

        _start_score = self.out_score_real_max if _mode_sort in ['descending', 'd'] else self.out_score_real_min
        _step = -1 if _mode_sort in ['descending', 'd'] else 1
        _step = _step * (self.out_score_real_max - self.out_score_real_min)/(len(self.raw_score_ratio_cum)-1)

        # _ts_list = []
        map_table = self.map_table
        real_min = map_table.query(col+'_count>0')['seg'].min()
        real_max = map_table.query(col+'_count>0')['seg'].max()
        for ri, row in map_table.iterrows():
            _seg = row['seg']
            _p = row[col + '_percent']
            y = None  # init out_score y = a * x + b

            if _seg == real_min:
                if _mode_ppt_score_min == 'defined':
                    y = self.out_score_real_min
            if _seg == real_max:
                if _mode_ppt_score_max == 'defined':
                    y = self.out_score_real_max
            if y is not None:
                row[col + '_ts'] = slib.round45r(y, self.out_decimal_digits)
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                result_ratio.append(format(_p, '.6f'))
                continue

            # loop: seeking ratio by percent to set out score
            for si, sr in enumerate(self.raw_score_ratio_cum):
                # sr == _p or sr > _p
                if (abs(sr - _p) < tiny_value) or (sr > _p):
                    if (abs(_p - sr) < tiny_value) or (si == 0):
                        y = _start_score + si*_step
                    elif _mode_prox == 'upper_min':
                        y = _start_score + si*_step
                    elif _mode_prox == 'lower_max':
                        if si > 0:
                            y = _start_score + (si - 1)*_step
                        else:
                            y = _start_score + si*_step
                    elif 'near' in _mode_prox:
                        if abs(_p-sr) < abs(_p-self.raw_score_ratio_cum[si-1]):
                            y = _start_score - si
                        elif abs(_p-sr) > abs(_p-self.raw_score_ratio_cum[si-1]):
                            y = _start_score + (si - 1)*_step
                        else:
                            if 'near_max' in _mode_prox:
                                y = _start_score + si*_step
                            else:
                                y = _start_score + (si - 1)*_step
                    else:
                        self.loginfo('Error Ratio Prox Mode: {}'.format(_mode_prox))
                        raise ValueError
                    break
            if y is not None:
                # self.loginfo('-1', row[col+'_ts'])
                row[col+'_ts'] = slib.round45r(y, self.out_decimal_digits)
                # self.loginfo('plt', row[col+'_ts'])
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                result_ratio.append(format(_p, '.6f'))
                # _ts_list.append(y)
            # end scanning raw_score_ratio_list
        # end scanning map_table

        self.result_formula_coeff = coeff_dict
        formula_dict = {k: '{cf[0][0]:.8f} * (x - {cf[1][1]:.0f}) + {cf[0][1]:.0f}'.format(cf=coeff_dict[k])
                        for k in coeff_dict}
        self.result_dict[col] = {
                                 'coeff': coeff_dict,
                                 'formula': formula_dict
                                 }
        self.result_ratio_dict[col] = dict()
        self.result_ratio_dict[col]['def'] = self.raw_score_ratio_cum
        self.result_ratio_dict[col]['dest'] = self.raw_score_ratio_cum
        self.result_ratio_dict[col]['match'] = result_ratio
        self.result_ratio_dict[col]['section'] = [coeff_dict[k][1] for k in coeff_dict.keys()]
        return True

    def get_formula_plt(self, field):
        # --step 1
        # claculate raw_score_endpoints
        # self.loginfo('   get raw score section ...')
        if not self.get_raw_section(field=field):
            return False

        # display ratio searching result at section i
        if self.logger:
            for i, (cumu_ratio, dest_ratio, match, raw_sec, out_sec) in \
                enumerate(zip(self.result_ratio_dict[field]['def'],
                              self.result_ratio_dict[field]['dest'],
                              self.result_ratio_dict[field]['match'],
                              self.result_ratio_dict[field]['section'],
                              self.out_score_section,
                              )):
                self.loginfo('   <{0:02d}> ratio: [def:{1:.4f}  dest:{2:.4f}  match:{3:.4f}] => '
                      'section_map: raw:[{4:3d}, {5:3d}] --> out:[{6:3d}, {7:3d}]'.
                      format(i + 1,
                             cumu_ratio,
                             dest_ratio,
                             match,
                             int(raw_sec[0]),
                             int(raw_sec[1]),
                             int(out_sec[0]),
                             int(out_sec[1])
                             )
                      )

        # --step 2
        # calculate Coefficients
        self.get_formula_coeff(field)
        self.result_dict[field] = {'coeff': copy.deepcopy(self.result_formula_coeff)}
        return True

    # -----------------------------------------------------------------------------------
    # formula-1: y = (y2-y1)/(x2 -x1)*(x - x1) + y1                   # a(x - b) + c
    #        -2:   = (y2-y1)/(x2 -x1)*x + (y1x2 - y2x1)/(x2 - x1)     # ax + b
    #        -3:   = [(y2-y1)*x + y1x2 - y2x1]/(x2 - x1)              # (ax + b) / c ; int / int
    def get_formula_coeff(self, field=None):

        # create raw score segments list
        # x_points = self.result_raw_endpoints
        # if self.strategy_dict['mode_section_point_start'] == 'share':
        #     step = 0
        # else:
        #     step = 1 if self.strategy_dict['mode_sort_order'] in ['ascending', 'a'] else -1
        # x_list = [(int(p[0])+(step if i > 0 else 0), int(p[1]))
        #           for i, p in enumerate(zip(x_points[:-1], x_points[1:]))]
        # # 3-problems: minus score,
        # #             less than min,
        # #             no ratio interval(not found because of last too large ratio!)
        # x_list = [(-1, -1)
        #           if p[0] < 0 or min(p) < self.raw_score_defined_min or (p[0]-p[1])*step > 0
        #           else p
        #           for p in x_list]

        x_list = self.result_ratio_dict[field]['section']

        # calculate coefficient
        y_list = self.out_score_section
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            v = x[1] - x[0]
            if v == 0:
                a = 0
                # mode_section_degraded
                _mode_section_degraded = self.strategy_dict['mode_section_degraded']
                if _mode_section_degraded == 'to_max':         # x1 == x2 : y = max(y1, y2)
                    b = max(y)
                elif _mode_section_degraded == 'to_min':       # x1 == x2 : y = min(y1, y2)
                    b = min(y)
                elif _mode_section_degraded == 'to_mean':      # x1 == x2 : y = mean(y1, y2)
                    b = np.mean(y)
                else:
                    self.loginfo('error mode_section_degraded value: {}'.format(_mode_section_degraded))
                    raise ValueError
            else:
                a = (y[1]-y[0])/v                   # (y2 - y1) / (x2 - x1)
                b = (y[0]*x[1]-y[1]*x[0])/v         # (y1x2 - y2x1) / (x2 - x1)
            self.result_formula_coeff.\
                update({i: [(a, b), (int(x[0]), int(x[1])), (int(y[0]), int(y[1]))]})
        return True

    # new at 2019-09-09
    # extract section points(first end point of first section and second point of all section) from map_table
    #   according ratios in preset ratio_list: raw_score_ratio_cum (cumulative ratio list)
    def get_raw_section(self, field):
        result_matched_ratio = []
        result_dest_ratio = []
        _ratio_cum_list = self.raw_score_ratio_cum

        section_real_min = self.df[field].min()
        section_real_max = self.df[field].max()
        section_defined_min = self.raw_score_defined_min
        section_defined_max = self.raw_score_defined_max

        _mode_cumu = self.strategy_dict['mode_ratio_cumu']
        _mode_order = self.strategy_dict['mode_sort_order']

        # first points of first section in raw score
        if self.strategy_dict['mode_section_point_first'] == 'real':
            section_start_point = section_real_min \
                if _mode_order in ['a', 'ascending'] else section_real_max
        else:
            section_start_point = section_defined_min \
                if _mode_order in ['a', 'ascending'] else section_defined_max

        result_section_point = [section_start_point]

        # ratio: preset,  percent: computed from data in map_table
        last_ratio = 0
        last_match = 0
        if self.strategy_dict['mode_section_point_start'] == 'share':
            _step = 0
        else:
            _step = 1 if _mode_order in ['a', 'ascending'] else -1
        for i, cumu_ratio in enumerate(_ratio_cum_list):
            this_seg_ratio = cumu_ratio-last_ratio
            dest_ratio = cumu_ratio if _mode_cumu == 'no' else this_seg_ratio + last_match
            result_dest_ratio.append(dest_ratio)

            # match percent by dest_ratio to get endpoint of this section from map_table
            this_section_end_point, this_match = \
                self.get_seg_from_map_table(field, dest_ratio)

            # last section at bottom
            if last_match >= 1:
                this_section_end_point = -1

            # save to result ratio
            result_matched_ratio.append(this_match)

            # save result endpoints (noshare, share)
            result_section_point.append(this_section_end_point)

            # set last point if mode_section_point_last == 'defined'
            if i > 1:
                # current point: item[i+1] == -1
                if (result_section_point[i+1] < 0) and (result_section_point[i] >= 0):
                    if self.strategy_dict['mode_section_point_last'] == 'defined':
                        if self.strategy_dict['mode_sort_order'] in ['d', 'descending']:
                            result_section_point[i] = section_defined_min
                        else:
                            result_section_point[i] = section_defined_max

            # save last segment endpoint and percent
            last_ratio = cumu_ratio
            last_match = this_match

        # step-2: process last same point
        for i in range(len(result_section_point)-1, 1, -1):
            if i == (len(result_section_point)-1):
                if result_section_point[i] == result_section_point[i-1]:
                    result_section_point[i] = -1
            else:
                if result_section_point[i+1] < 0:
                    if result_section_point[i] == result_section_point[i - 1]:
                        result_section_point[i] = -1
        # print(result_section_point)

        # set last point again if [-1] >= 0 else alread set in loop
        if self.strategy_dict['mode_section_point_last'] == 'defined':
            for i in range(len(result_section_point)-1, 0, -1):
                if result_section_point[i] >= 0:
                    if self.strategy_dict['mode_sort_order'] in ['d', 'descending']:
                        result_section_point[i] = section_defined_min
                    else:
                        result_section_point[i] = section_defined_max
                    break

        # create section
        make_section = []
        i = 0
        sectio_lost = False
        for x, y in zip(result_section_point[:-1],
                        result_section_point[1:]):
            _step = -1 if self.strategy_dict['mode_sort_order'] in ['d', 'descending'] else 1
            _y = y
            _x = None
            if x != y:
                if self.strategy_dict['mode_section_point_start'] == 'share':
                    if not sectio_lost:
                        _x = x
                    else:
                        if x > 0 and y > 0:
                            sectio_lost = False
                            _x = x + _step
                        else:
                            _x, _y = -1, -1
                else:
                    _x = x + _step if i > 0 else x
            else:
                if i > 0:
                    _x, _y = -1, -1
                else:
                    _x = x
                sectio_lost = True
            _x, _y = (_x, _y) if (_x >= 0) and (_y >= 0) else (-1, -1)
            if self.strategy_dict['mode_section_lost'] == 'zip':
                if (_x < 0) or (_y < 0):
                    continue
            make_section.append((_x, _y))
            i += 1
        # print(make_section)
        # add [-1, -1] to tail, fill up that removed in make_section because 'zip'
        len_less = len(self.raw_score_ratio_cum) - len(make_section)
        if len_less > 0:
            make_section += [[-1, -1] for _ in range(len_less)]

        self.result_ratio_dict[field] = {
            'def': self.raw_score_ratio_cum,
            'dest': result_dest_ratio,
            'match': result_matched_ratio,
            'section': make_section
            }

        self.result_raw_endpoints = result_section_point

        return True

    # new at 2019-09-09
    def get_seg_from_map_table(self, field, dest_ratio):

        _mode_prox = self.strategy_dict['mode_ratio_prox']
        _top_index = self.map_table.index.max()
        _tiny = self.tiny_value

        _seg = -1
        _percent = -1
        last_percent = -1
        last_seg = None
        dist_to_last = 1000
        use_last_seg = False
        for index, row in self.map_table.iterrows():
            _percent = row[field+'_percent']
            _seg = row['seg']
            dist_to_this = abs(_percent - dest_ratio)

            # at table bottom or lowest score, use_current
            # if (index == _top_index) or (_percent >= 1):
            #     break

            # this >= dest_ratio
            if _percent >= dest_ratio:
                # top: single point section
                if last_seg is None:
                    break

                # ratio_prox
                if 'near' in _mode_prox:
                    # same dist
                    if abs(dist_to_this-dist_to_last) < _tiny:
                        if _mode_prox == 'near_min':
                            use_last_seg = True
                        else:
                            use_last_seg = False
                    elif dist_to_this < dist_to_last:
                        use_last_seg = False
                    else:
                        use_last_seg = True
                elif _mode_prox == 'lower_max':
                    # this != dest_ratio
                    if dist_to_this > _tiny:
                        use_last_seg = True
                elif _mode_prox == 'upper_min':
                    # last_ratio == dest_ratio
                    if dist_to_last < _tiny:
                        use_last_seg = True
                else:
                    self.loginfo('Error ratio prox mode: {}'.format(_mode_prox))
                    raise ValueError
                break
            dist_to_last = dist_to_this
            last_seg = _seg
            last_percent = _percent

        if use_last_seg:
            return last_seg, last_percent
        return _seg, _percent

    # create report and col_ts in map_table
    def make_report(self):
        self.out_report_doc = '{}[{}]\n'.\
            format('Transform Model: '.rjust(20), self.model_name)
        self.out_report_doc += '{}{}\n'.\
            format('running-time: '.rjust(20), time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime()))
        self.out_report_doc += '---'*40 + '\n'
        self.out_report_doc += format('strategies: ', '>20') + '\n'

        # self.loginfo('   create report ...')
        for k in self.strategy_dict.keys():
            self.out_report_doc += ' ' * 20 + '{0:<30s} {1}'. \
                format(k + ' = ', self.strategy_dict[k]) + '\n'
        self.out_report_doc += '---'*40 + '\n'
        for col in self.cols:
            self.out_report_doc += self.make_field_report(col)

    def make_field_report(self, field=''):
        score_dict = {x: y for x, y in zip(self.map_table['seg'], self.map_table[field+'_count'])}
        p = 0 if self.strategy_dict['mode_sort_order'] in ['ascending', 'a'] else 1
        self.result_formula_text_list = []
        _fi = 1
        for k in self.result_dict[field]['coeff']:
            formula = self.result_dict[field]['coeff'][k]

            _break = True
            _step = 1 if formula[1][0] < formula[1][1] else -1
            for _score in range(int(formula[1][0]), int(formula[1][1])+_step, _step):
                if score_dict.get(_score, -1) > 0:
                    _break = False
                    break

            if _break:
                continue

            if formula[0][0] != 0:
                self.result_formula_text_list += \
                    ['(section-{0:3d}):  y = {1:0.8f}*(x-{2:2d}) + {3:2d}'.
                     format(int(_fi), formula[0][0], int(formula[1][p]), int(formula[2][p]))]
            else:
                if formula[2][0] != formula[2][1]:
                    self.result_formula_text_list += \
                        ['(section-{0:3d}):  y = {1:0.8f}*(x-{2:3d}) + {3}({4:3d}, {5:3d})'.
                         format(int(_fi),
                                formula[0][0], formula[1][p],
                                self.strategy_dict['mode_section_degraded'],
                                formula[2][0], formula[2][1])
                         ]
                # y2 == y1
                else:
                    self.result_formula_text_list += \
                        ['(section-{0:3d}):  y = 0 * (x-{2:5.2f}) + {3:5.2f}'.
                         format(_fi,
                                formula[0][0],
                                formula[1][p],
                                formula[2][0])
                         ]
            _fi += 1

        # report start
        # tiltle
        field_title = '<< score field: [{}] >>\n' + '- -'*40 + '\n'
        _out_report_doc = field_title.format(field)

        # calculating for ratio and segment
        plist = self.raw_score_ratio_cum
        if self.out_decimal_digits == 0:
            _out_report_doc += '  raw score def ratio: [{}]\n'.\
                format(', '.join([format(x, '10.6f') for x in self.raw_score_ratio_cum]))
            _out_report_doc += '           dest ratio: [{}]\n'.\
                format(', '.join([format(float(x), '10.6f')
                                  for x in self.result_ratio_dict[field]['dest']]))
            _out_report_doc += '          match ratio: [{}]\n'. \
                format(', '.join([format(float(x), '10.6f')
                                  for x in self.result_ratio_dict[field]['match']]))
        else:
            _out_report_doc += '  raw score sec ratio: [{}]\n'.\
                format(', '.join([format(plist[j]-plist[j-1] if j > 0 else plist[0], '16.6f')
                        for j in range(len(plist))]))
            _out_report_doc += '           cumu ratio: [{}]\n'.\
                format(', '.join([format(x, '16.6f') for x in self.raw_score_ratio_cum]))
            _out_report_doc += '          match ratio: [{}]\n'.\
                format(', '.join([format(float(x), '16.6f') for x in self.result_ratio_dict[field]['match']]))

        # get raw segment from result_dict
        _raw_seg_list = [c[1] for c in self.result_dict[field]['coeff'].values()]
        if self.out_decimal_digits == 0:
            _out_report_doc += '              section: [{}]\n'.\
                format(', '.join(['({:3d}, {:3d})'.format(int(x), int(y)) for x, y in _raw_seg_list]))
        else:
            _out_report_doc += '              section: [{}]\n'.\
                format(', '.join(['({:6.2f}, {:6.2f})'.format(x, y) for x, y in _raw_seg_list]))

        # get out segment from result_dict[]['coeff']
        _out_seg_list = [x[2] for x in self.result_dict[field]['coeff'].values()]
        if self.out_decimal_digits > 0:
            _out_report_doc += '  out  score  section: [{}]\n'.\
                format(', '.join(['({:6.2f}, {:6.2f})'.format(x, y) for x, y in _out_seg_list]))
        else:
            _out_report_doc += '  out  score  section: [{}]\n'.\
                format(', '.join(['({:>3.0f}, {:>3.0f})'.format(x, y) for x, y in _out_seg_list]))

        # transforming formulas
        _out_report_doc += '- -'*40 + '\n'
        for i, col in enumerate(self.result_formula_text_list):
            if i == 0:
                _out_report_doc += 'transforming formulas:\n'
            _out_report_doc += '                       {}\n'.format(col)

        # statistics for raw and out score
        _out_report_doc += '- -'*40 + '\n'
        _out_report_doc += format('statistics:', '>22s')

        # raw score data describing
        _max, _min, __mean, _median, _mode, __std, _skew, _kurt = \
            self.df[field].max(),\
            self.df[field].min(),\
            self.df[field].mean(),\
            self.df[field].median(), \
            self.df[field].mode()[0], \
            self.df[field].std(),\
            self.df[field].skew(),\
            sts.kurtosis(self.df[field], fisher=False)
        _out_report_doc += ' raw: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                           format(_max, _min, __mean, _median, _mode)
        _out_report_doc += ' '*28 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                           format(__std, __std/__mean, _max-_min, _skew, _kurt)

        # _count_zero = self.map_table.query(field+'_count==0')['seg'].values
        _count_non_zero = self.map_table.groupby('seg')[[field+'_count']].sum().query(field+'_count>0').index
        _count_zero = [x for x in range(self.raw_score_defined_min, self.raw_score_defined_max+1)
                       if x not in _count_non_zero]
        _out_report_doc += ' '*28 + 'empty_value={}\n' .\
                           format(slib.set_ellipsis_in_digits_sequence(_count_zero))

        # out score data describing
        _max, _min, __mean, _median, _mode, __std, _skew, _kurt = \
            self.outdf[field+'_ts'].max(),\
            self.outdf[field+'_ts'].min(),\
            self.outdf[field+'_ts'].mean(),\
            self.outdf[field+'_ts'].median(), \
            self.outdf[field+'_ts'].mode()[0],\
            self.outdf[field+'_ts'].std(),\
            self.outdf[field+'_ts'].skew(), \
            sts.kurtosis(self.outdf[field+'_ts'], fisher=False)
        _out_report_doc += ' '*23 + 'out: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                           format(_max, _min, __mean, _median, _mode)
        _out_report_doc += ' '*28 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                           format(__std, __std/__mean, _max-_min, _skew, _kurt)
        # _count_zero = self.map_table.query(field+'_count==0')[field+'_ts'].values
        _count_non_zero = self.map_table.groupby(field+'_ts')[[field+'_count']].sum().query(field+'_count>0').index
        if self.model_type == 'ppt': # msin.Models[self.model_name].type == 'plt':
            _count_zero = [x for x in range(int(self.out_score_real_min), int(self.out_score_real_max) + 1)
                           if x not in _count_non_zero]
        else:
            _count_zero = ''
        _out_report_doc += ' '*28 + 'empty_value={}\n' .\
                           format(slib.set_ellipsis_in_digits_sequence(_count_zero))
        _out_report_doc += 'count: '.rjust(28) + 'record={}\n' .\
                           format(self.outdf.count()[0])

        # differece between raw and out score
        _out_report_doc += '- -'*40 + '\n'
        _diff_raw_out = self.outdf[field+'_ts']-self.outdf[field]
        _out_report_doc += ' score shift(out-raw):' \
                           ' shift_max={:3.1f}' \
                           '  shift_min={:3.1f}' \
                           '  shift_down_percent={:.2f}%\n'.\
                           format(
                                  max(_diff_raw_out),
                                  min(_diff_raw_out),
                                  _diff_raw_out[_diff_raw_out < 0].count()/_diff_raw_out.count()*100
                              )
        _diff_list = []
        for coeff in self.result_formula_coeff.values():
            rseg = coeff[1]
            oseg = coeff[2]
            a = coeff[0][0]
            b = coeff[0][1]
            if rseg[0] < 0:
                continue
            # print(rseg, oseg)
            if rseg[0] >= oseg[0]:
                if rseg[1] > oseg[1]:
                    _diff_list.append(rseg)
            if (rseg[0] > oseg[0]) and (rseg[1] <= oseg[1]):
                _diff_list.append((int(rseg[0]), int(slib.round45r(b / (1 - a)))))
            if (rseg[0] < oseg[0]) and (rseg[1] >= oseg[1]):
                _diff_list.append((int(slib.round45r(b / (1 - a), 0)), int(rseg[1])))
        _out_report_doc += '   shift down segment: ' + str(_diff_list) + ' => '
        # merge to some continuous segments
        while True:
            _diff_loop = False
            for i in range(len(_diff_list)-1):
                if abs(_diff_list[i][1]-_diff_list[i+1][0]) == 1:
                    _diff_list[i] = (_diff_list[i][0], _diff_list[i+1][1])
                    _diff_list.pop(i+1)
                    _diff_loop = True
                    break
            if not _diff_loop:
                break
        _diff_list = [x for x in _diff_list if x[0] != x[1]]
        _out_report_doc += str(_diff_list) + '\n'
        _out_report_doc += '---'*40 + '\n'

        return _out_report_doc

    def report(self):
        print(self.out_report_doc)

    def plot(self, mode='model'):
        plot_name = ['raw', 'out', 'model', 'shift', 'dist', 'diff', 'bar', 'rawbar', 'outbar']
        if mode not in plot_name:
            self.loginfo('error plot: [{}] not in {}'.format(mode, plot_name))
            return
        if mode in 'shift, model':
            # mode: model describe the differrence of input and output score.
            self.plot_model()
        elif mode in 'dist':
            self.plot_dist_seaborn()
        elif mode in 'bar':
            self.plot_bar('all')
        elif mode == 'rawbar':
            self.plot_bar('raw')
        elif mode == 'outbar':
            self.plot_bar('out')
        elif mode in 'diff':
            self.plot_diff()
        elif not super(PltScore, self).plot(mode):
            self.loginfo('error plot: [{}] is invalid!'.format(mode))

    def plot_diff(self):
        x = [int(x) for x in self.map_table['seg']][::-1]   # np.arange(self.mode_score_paper_max+1)
        raw_label = [str(x) for x in self.map_table['seg']][::-1]
        for f in self.cols:
            df = [v if self.map_table.query('seg=='+str(v))[f+'_count'].values[0] > 0 else 0 for v in x]
            outdf = list(self.map_table[f + '_ts'])[::-1]
            outdf = [out if raw > 0 else 0 for raw, out in zip(df, outdf)]
            # fig1 = plot.figure('subject: '+f)
            fig, ax = plot.subplots()
            # ax.set_figure(fig1)
            ax.set_title(self.model_name+'['+f+']: diffrence between raw and out')
            ax.set_xticks(x)
            ax.set_xticklabels(raw_label)
            width = 0.4
            bar_wid = [p - width/2 for p in x]
            rects1 = ax.bar(bar_wid, df, width, label=f)
            bar_wid = [p + width/2 for p in x]
            rects2 = ax.bar(bar_wid, outdf, width, label=f+'_ts')

            """Attach a text label above each bar in *rects*, displaying its height."""
            for i, rects in enumerate([rects1, rects2]):
                for rect in rects:
                    if i == 0:
                        notes = rect.get_height()
                    else:
                        if rect.get_height() > 0:
                            notes = rect.get_height() - rect.get_x()
                        else:
                            notes = 0
                    height = rect.get_height()
                    ax.annotate('{}'.format(int(notes)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            ax.legend(loc='upper left', shadow=True, fontsize='x-large')
            fig.tight_layout()
            plot.show()

    # def plot_rawbar(self):
    #     raw_label = [str(x) for x in range(self.mode_score_paper_max+1)]
    #     x_data = list(range(self.mode_score_paper_max+1))
    #     seg_list = list(self.map_table.seg)
    #     for f in self.cols:
    #         df = [self.map_table.query('seg=='+str(xv))[f+'_count'].values[0]
    #                     if xv in seg_list else 0
    #                     for xv in x_data]
    #         fig, ax = plot.subplots()
    #         ax.set_title(self.model_name+'['+f+']: bar graph')
    #         ax.set_xticks(x_data)
    #         ax.set_xticklabels(raw_label)
    #         width = 0.8
    #         bar_wid = [p - width/2 for p in x_data]
    #         ax.bar(bar_wid, df, width, label=f)

    def plot_outbar(self):
        x_label = [str(x) for x in range(self.out_score_real_max + 1)]
        x_data = list(range(self.out_score_real_max + 1))
        for f in self.cols:
            out_ = self.outdf.groupby(f+'_ts').count()[f]
            outdf = [out_[int(v)] if int(v) in out_.index else 0 for v in x_label]
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+'_out_score]: bar graph')
            ax.set_xticks(x_data)
            ax.set_xticklabels(x_label)
            width = 0.8
            bar_wid = [p - width/2 for p in x_data]
            ax.bar(bar_wid, outdf, width, label=f)

    def plot_bar(self, display='all', hcolor='r', hwidth=6):
        raw_label = [str(x) for x in range(int(self.out_score_real_max) + 1)]
        x_data = list(range(int(self.out_score_real_max) + 1))
        seg_list = list(self.map_table.seg)
        for f in self.cols:
            df = [self.map_table.query('seg=='+str(xv))[f+'_count'].values[0]
                        if xv in seg_list else 0
                        for xv in x_data]
            out_ = self.outdf.groupby(f+'_ts').count()[f]
            outdf = [out_[int(v)] if int(v) in out_.index else 0 for v in raw_label]
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: bar graph')
            ax.set_xticks(x_data)
            ax.set_xticklabels(raw_label)
            width = 0.4
            bar_wid = [p + width/2 for p in x_data]
            if display in ['all']:
                raw_bar = ax.bar(bar_wid, df, width, label=f)
                out_bar = ax.bar(bar_wid, outdf, width, label=f + '_ts')
                disp_bar =[raw_bar, out_bar]
            elif 'raw' in display:
                raw_bar = ax.bar(bar_wid, df, width, label=f)
                disp_bar = [raw_bar]
                ax.set_title(self.model_name+'[{}]  mean={:.2f}, std={:.2f}, max={:3d}'.
                             format(f, self.df[f].mean(), self.df[f].std(), self.df[f].max()))
            else:
                out_bar = ax.bar(bar_wid, outdf, width, label=f + '_ts')
                disp_bar = [out_bar]
                ax.set_title(self.model_name + '[{}]  mean={:.2f}, std={:.2f}, max={:3d}'.
                             format(f, self.outdf[f].mean(),
                                    self.outdf[f+'_ts'].std(), self.outdf[f+'_ts'].max()))
            for bars in disp_bar:
                make_color = 0
                last_height = 0
                for _bar in bars:
                    height = _bar.get_height()
                    xpos = _bar.get_x() + _bar.get_width() / 2
                    note_str= '{}'.format(int(height))
                    ypos = 0
                    if (height > 100) and abs(height - last_height) < 20:
                        if height < last_height:
                            ypos = - 10
                        else:
                            ypos = + 10
                    ax.annotate(note_str,
                                xy=(xpos, height),
                                xytext=(0, ypos),              # vertical offset
                                textcoords="offset points",
                                ha='center',
                                va='bottom'
                                )
                    if display == 'all':
                        continue
                    if make_color == 2:
                        plot.plot([xpos, xpos], [0, height], hcolor, linewidth=hwidth)
                        make_color = 0
                    else:
                        make_color += 1
                    last_height = height + ypos
            if display == 'all':
                ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            fig.tight_layout()
            plot.show()


    def plot_rawbar(self, hcolor='r', hwidth=6):
        raw_label = reversed([str(x) for x in self.map_table.seg])
        x_data = list(range(self.raw_score_defined_max + 1))
        for f in self.cols:
            df = [self.map_table.query('seg=='+str(xv))[f+'_count'].values[0]
                  if xv in self.map_table.seg else 0
                  for xv in x_data]

            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: bar graph')
            ax.set_xticks(x_data)
            ax.set_xticklabels(raw_label)
            width = 0.4
            bar_wid = [p + width/2 for p in x_data]

            raw_bar = ax.bar(bar_wid, df, width, label=f)
            disp_bar = [raw_bar]
            ax.set_title(self.model_name+'[{}]  mean={:.2f}, std={:.2f}, max={:3d}'.
                         format(f, self.df[f].mean(), self.df[f].std(), self.df[f].max()))

            for bars in disp_bar:
                make_color = 0
                last_height = 0
                for _bar in bars:
                    height = _bar.get_height()
                    xpos = _bar.get_x() + _bar.get_width() / 2
                    note_str = '{}'.format(int(height))
                    ypos = 0
                    if (height > 100) and abs(height - last_height) < 20:
                        if height < last_height:
                            ypos = -10
                        else:
                            ypos = +10
                    ax.annotate(note_str,
                                xy=(xpos, height),
                                xytext=(0, ypos),              # vertical offset
                                textcoords="offset points",
                                ha='center',
                                va='bottom'
                                )
                    if make_color == 2:
                        plot.plot([xpos, xpos], [0, height], hcolor, linewidth=hwidth)
                        make_color = 0
                    else:
                        make_color += 1
                    last_height = height + ypos
            fig.tight_layout()
            plot.show()


    def plot_dist(self):
        def plot_dist_fit(field, _label):
            x_data = self.outdf[field]
            # local var _mu, __std
            _mu = np.mean(x_data)
            __std = np.std(x_data)
            count, bins, patches = ax.hist(x_data, 35)
            x_fit = ((1 / (np.sqrt(2 * np.pi) * __std)) * np.exp(-0.5 * (1 / __std * (bins - _mu))**2))
            # x_fit = x_fit * max(count)/max(x_fit)
            _color = 'y--' if '_ts' in field else 'g--'
            ax.plot(bins, x_fit, _color, label=_label)
            ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            # print(field, len(count), sum(count), count)
        for f in self.cols:
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: distribution garph')
            # fit raw score distribution
            plot_dist_fit(f, 'raw score')
            # fit out score distribution
            plot_dist_fit(f+'_ts', 'out score')
        plot.show()

    def plot_dist_seaborn(self):
        for f in self.cols:
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: distribution garph')
            sbn.kdeplot(self.outdf[f], shade=True)
            sbn.kdeplot(self.outdf[f+'_ts'], shade=True)

    def plot_model(self, down_line=True):
        # 分段线性转换模型
        plot.rcParams['font.sans-serif'] = ['SimHei']
        plot.rcParams.update({'font.size': 8})
        for i, col in enumerate(self.cols):
            # result = self.result_dict[col]['coeff']
            # raw_points = [result[x][1][0] for x in result] + [result[max(result.keys())][1][1]]
            in_max = max(self.result_raw_endpoints)
            in_min = min(self.result_raw_endpoints)
            out_min = min([min(p) for p in self.out_score_section])
            out_max = max([max(p) for p in self.out_score_section])

            plot.figure(col+'_ts')
            plot.rcParams.update({'font.size': 10})
            plot.title(u'转换模型({})'.format(col))
            plot.xlim(in_min, in_max)
            plot.ylim(out_min, out_max)
            plot.xlabel(u'\n\n原始分数')
            plot.ylabel(u'转换分数')
            plot.xticks([])
            plot.yticks([])

            formula = self.result_dict[col]['coeff']
            for cfi, cf in enumerate(formula.values()):
                # segment map function graph
                _score_order = self.strategy_dict['mode_sort_order']
                x = cf[1] if _score_order in ['ascending', 'a'] else cf[1][::-1]
                y = cf[2] if _score_order in ['ascending', 'a'] else cf[2][::-1]
                plot.plot(x, y, linewidth=2)

                # line from endpoint to axis
                for j in [0, 1]:
                    plot.plot([x[j], x[j]], [0, y[j]], '--', linewidth=1)
                    plot.plot([0, x[j]], [y[j], y[j]], '--', linewidth=1)
                for j, xx in enumerate(x):
                    plot.text(xx-2 if j == 1 else xx, out_min-2, '{}'.format(int(xx)))

                # out_score scale value beside y-axis
                for j, yy in enumerate(y):
                    plot.text(in_min+1, yy+1 if j == 0 else yy-2, '{}'.format(int(yy)))
                    if y[0] == y[1]:
                        break

            if down_line:
                # darw y = x for showing score shift
                plot.plot((0, in_max), (0, in_max), 'r--', linewidth=2, markersize=2)

        plot.show()
        return
