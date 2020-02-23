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
import abc
# import fractions as fr
# import bisect as bst
# import array


# external import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as sts
# import seaborn as sbn

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
    def run(self):
        """
        运行转换处理过程
        run to get maptable, formula, outdf...
        """

    @abc.abstractmethod
    def report(self):
        """
        返回系统生成的分数转换报告
        return score transforming report created by model running
        """

    @abc.abstractmethod
    def plot(self, mode='raw'):
        """
        显示原始分数、转换模型、转换结果的可视化图形
        plot graphs for raw score，model and result score
        """

    def save_outdf_to_csv(self, filename):
        self.outdf.to_csv(filename, index=False, encoding='utf-8')

    def save_report_doc(self, filename):
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write(self.out_report_doc)

    def save_maptable_doc(self, filename, col_width=20):
        """
        保存分数转换映射表为文档文件
        save map talbe to text doc file
        # deprecated: use module ptt to create griding and  paging text doc
        # with open(filename, mode='w', encoding='utf-8') as f:
        #     f.write(ptt.make_mpage(self.maptable, page_line_num=50))
        """
        with open(filename, mode='w', encoding='utf-8') as f:
            t = ' '
            for cname in self.maptable.columns:
                t += ('{}'.format(cname)).rjust(col_width)
            t += '\n'
            start = False
            for row_no, row in self.maptable.iterrows():
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
        # super(PltScore, self).__init__(model_name, model_type)

        # model
        self.model_name = model_name
        self.model_type = model_type

        # raw data
        self.df = pd.DataFrame()
        self.cols = []

        # new properties
        self.__raw_score_ratio_cum = []
        self.__raw_score_real_max = None
        self.__raw_score_real_min = None
        self.__raw_score_defined_max = None
        self.__raw_score_defined_min = None
        self.__out_score_section = []
        self.__out_score_max = None
        self.__out_score_min = None

        # strategy
        self.__strategy_dict = dict()

        # run control
        self.__display=True
        self.__logger=None

        # calc precision
        self.__value_tiny_value = 10 ** -12
        self.__value_out_score_decimals = 0

        # result
        self.maptable = pd.DataFrame()
        self.outdf = pd.DataFrame()
        self.raw_section = []
        self.result_dict = dict()
        self.__report_doc = ''
        self.__result_raw_endpoints = []
        self.__result_ratio_dict = dict()
        self.__result_formula_coeff = dict()
        self.__result_formula_text_list = ''

    # plt
    def set_data(self, df=None, cols=None):

        # check and set rawdf
        if type(df) == pd.Series:
            self.df = pd.DataFrame(df)
        elif type(df) == pd.DataFrame:
            self.df = df
        else:
            if self.__logger:
                self.__logger.loginfo('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set outdf
        if type(cols) is str:
            if cols in self.df.columns:
                self.cols = [cols]
                return True
            else:
                if self.__logger:
                    self.__logger.loginfo('invalid field name in cols: {}'.format(cols))
        elif type(cols) not in [list, tuple]:
            self.__logger.loginfo('col set fail!\n not a list or tuple!')
            return False
        elif sum([1 if sf in df else 0 for sf in cols]) != len(cols):
            self.__logger.loginfo('field of cols not in rawdf.columns!')
            return False
        else:
            self.cols = cols
            return True

    def set_para(self,
                 raw_score_ratio=None,
                 out_score_section=None,
                 raw_score_defined_min=0,
                 raw_score_defined_max=100,
                 mode_section_point_ratio_prox='upper_min',
                 mode_section_point_ratio_cumu='no',
                 mode_score_sort_order='descending',
                 mode_section_point_first='real',
                 mode_section_point_start='step',
                 mode_section_point_last='real',
                 mode_section_degraded='to_max',
                 mode_section_lost='real',
                 mode_score_zero='real',
                 value_out_score_decimals=None,
                 value_tiny_value=10**-8,
                 logger=None,
                 display=True,
                 ):
        if isinstance(display, bool):
            self.__display=display

        self.__value_out_score_decimals = value_out_score_decimals

        if mode_score_sort_order in ['descending', 'd']:
            raw_p = raw_score_ratio
            self.__out_score_section = out_score_section
        else:
            raw_p = raw_score_ratio[::-1]
            self.__out_score_section = tuple(x[::-1] for x in out_score_section[::-1])
        ratio_sum = sum(raw_p)
        self.__raw_score_ratio_cum = [sum(raw_p[0:x + 1]) / ratio_sum for x in range(len(raw_p))]

        self.__raw_score_defined_min, self.__raw_score_defined_max = raw_score_defined_min, raw_score_defined_max

        self.__strategy_dict['mode_section_point_ratio_prox'] = mode_section_point_ratio_prox
        self.__strategy_dict['mode_section_point_ratio_cumu'] = mode_section_point_ratio_cumu
        self.__strategy_dict['mode_score_sort_order'] = mode_score_sort_order
        self.__strategy_dict['mode_section_point_first'] = mode_section_point_first
        self.__strategy_dict['mode_section_point_start'] = mode_section_point_start
        self.__strategy_dict['mode_section_point_last'] = mode_section_point_last
        self.__strategy_dict['mode_section_degraded'] = mode_section_degraded
        self.__strategy_dict['mode_section_lost'] = mode_section_lost

        self.__value_tiny_value = value_tiny_value
        self.__logger = logger

    def __loginfo(self, ms):
        if self.__logger:
            self.__logger.loginfo(ms)
        else:
            if self.__display:
                print(ms)
    # --------------data and para setting end

    # plt score run
    def run(self):

        self.__logger.log('stm start ...\n' + '-' * 120, 'debug')
        stime = time.time()

        if self.__out_score_section is not None:
            self.__out_score_real_max = max([max(x) for x in self.__out_score_section])
            self.__out_score_real_min = min([min(x) for x in self.__out_score_section])

        # calculate seg table
        # self.logger.loginfo('calculating maptable ...')
        _segsort = 'a' if self.__strategy_dict['mode_score_sort_order'] in ['ascending', 'a'] else 'd'
        seg_model = slib.get_segtable(
                  df=self.df,
                  cols=self.cols,
                  segmax=self.__raw_score_defined_max,
                  segmin=self.__raw_score_defined_min,
                  segsort=_segsort,
                  segstep=1,
                  display=False,
                  usealldata=False
                  )
        self.maptable = seg_model.outdf   # .copy(deep=True)

        # create field_fr in maptable
        #   strange error!!: some seg percent to zero
        #   self.maptable[f+'_percent'] = self.maptable[f+'_fr'].apply(lambda x: float(x))
        # for f in self.cols:
        #     max_sum = max(self.maptable[f+'_sum'])
        #     max_sum = 1 if max_sum == 0 else max_sum
        #     self.maptable[f+'_fr'] = \
        #         self.maptable[f+'_sum'].apply(lambda x: fr.Fraction(x, max_sum))
            # self.maptable.astype({f+'_fr': fr.Fraction})     # encounter error in python 3.7.4

        # transform score on each field
        self.result_dict = dict()
        self.outdf = self.df.copy(deep=True)
        for i, col in enumerate(self.cols):
            self.__logger.loginfo('transform score: {0} => {0}_ts'.format(col) + '\n' + '-' * 120)

            # there is a problem: max set to locale value to each col
            self.__raw_score_real_max = self.df[col].max()
            self.__raw_score_real_min = self.df[col].min()

            # get formula and save
            _get_formula = False
            if self.model_type == 'ppt': # msin.Models[self.model_name].type == 'ppt':
                # self.logger.loginfo('get ppt formula ...')
                _get_formula = self.__get_formula_ppt(col)
            elif self.model_type == 'plt':
                # self.logger.loginfo('get plt formula ...')
                _get_formula = self.__get_formula_plt(col)
            else:
                self.__logger.log('error model type: not supported type={}'.format(self.model_type),
                                     'error')
                return None
            if not _get_formula:
                self.__logger.log('error calc: getting formula fail !', 'error')
                return None

            # get field_ts in outdf
            self.outdf.loc[:, (col + '_ts')] = \
                self.outdf[col].apply(
                    lambda x: self.__ts_formula(col, x))

        # create col_ts in maptable
        df_map = self.maptable
        for col in self.cols:
            # self.logger.loginfo('   calculate: maptable[{0}] => [{0}_ts]'.format(col))
            col_name = col + '_ts'
            df_map.loc[:, col_name] = df_map['seg'].apply(
                lambda x: self.__ts_formula(col, x))

        # make report doc
        self.__make_report()

        self.__logger.log('transform score end, elapsed-time:{:.4f}'.format(time.time() - stime) + '\n' + '-' * 120,
                             'debug')
        self.__logger.loginfo(self.__report_doc)

    # run end

    def __ts_formula(self, field, x):
        # -----------------------------------------------------------------------------------
        # formula for higher precise, int/int to float
        # original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
        # used to : y = (a*x + b) / c
        #           a=(y2-y1)
        #           b=y1x2-y2x1
        #           c=(x2-x1)
        if x > self.__raw_score_defined_max:
            return self.__out_score_real_max
        if x < self.__raw_score_defined_min:
            return self.__out_score_real_min
        for cf in self.result_dict[field]['coeff'].values():
            if (cf[1][0] <= x <= cf[1][1]) or (cf[1][0] >= x >= cf[1][1]):
                a = (cf[2][1] - cf[2][0])
                b = cf[2][0] * cf[1][1] - cf[2][1] * cf[1][0]
                c = (cf[1][1] - cf[1][0])
                # x1 == x2: use mode_section_degraded: max, min, mean(y1, y2)
                if c == 0:
                    if self.__strategy_dict['mode_section_degraded'] == 'to_max':
                        return slib.round45(max(cf[2]), self.__value_out_score_decimals)
                    elif self.__strategy_dict['mode_section_degraded'] == 'to_min':
                        return slib.round45(min(cf[2]), self.__value_out_score_decimals)
                    elif self.__strategy_dict['mode_section_degraded'] == 'to_mean':
                        return slib.round45(np.mean(cf[2]))
                    else:
                        # invalid mode
                        return -10000
                return slib.round45((a * x + b) / c, self.__value_out_score_decimals)
        return -10000

    # formula for b900, b300
    # coeff:   (a=0, b=x), (x, x), (y, y))
    # formula: y = 0*x + y for [x, x] --> [y, y]
    def __get_formula_ppt(self, col):
        self.__result_raw_endpoints = [x for x in self.maptable['seg']]
        self.maptable.loc[:, col+'_ts'] = -1
        coeff_dict = dict()
        section_index_dict = dict()
        result_ratio = []
        value_tiny_value = self.__value_tiny_value     # used to judge zero(s==0) or equality(s1==s2)

        _mode_sort = self.__strategy_dict['mode_score_sort_order']

        if _mode_sort in ['d', 'descending']:
            _mode_ppt_score_min = self.__strategy_dict['mode_section_point_last']  # real_min or paper_min
            _mode_ppt_score_max = self.__strategy_dict['mode_section_point_first']  # real_max or paper_max
        else:
            _mode_ppt_score_max = self.__strategy_dict['mode_section_point_last']  # real_min or paper_min
            _mode_ppt_score_min = self.__strategy_dict['mode_section_point_first']  # real_max or paper_max

        _mode_prox = self.__strategy_dict['mode_section_point_ratio_prox']

        _start_score = self.__out_score_real_max if _mode_sort in ['descending', 'd'] else self.__out_score_real_min
        _step = -1 if _mode_sort in ['descending', 'd'] else 1
        _step = _step * (self.__out_score_real_max - self.__out_score_real_min) / (len(self.__raw_score_ratio_cum) - 1)

        # _ts_list = []
        maptable = self.maptable
        real_min = maptable.query(col+'_count>0')['seg'].min()
        real_max = maptable.query(col+'_count>0')['seg'].max()
        for ri, row in maptable.iterrows():
            _seg = row['seg']
            _p = row[col + '_percent']
            y = None  # init out_score y = a * x + b

            if _seg == real_min:
                if _mode_ppt_score_min == 'defined':
                    y = self.__out_score_real_min
            if _seg == real_max:
                if _mode_ppt_score_max == 'defined':
                    y = self.__out_score_real_max
            if y is not None:
                row[col + '_ts'] = slib.round45(y, self.__value_out_score_decimals)
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                result_ratio.append(format(_p, '.6f'))
                continue

            # loop: seeking ratio by percent to set out score
            y_pos = None
            for si, sr in enumerate(self.__raw_score_ratio_cum):
                # sr == _p or sr > _p
                y_pos = si
                if (abs(sr - _p) < value_tiny_value) or (sr > _p):
                    # top(si==0) or equal(p==sr)
                    if (abs(_p - sr) < value_tiny_value) or (si == 0):
                        y = _start_score + si*_step
                    elif _mode_prox == 'upper_min':
                        y = _start_score + si*_step
                    elif _mode_prox == 'lower_max':
                        # si > 0
                        y = _start_score + (si - 1)*_step
                        y_pos = si - 1
                    elif 'near' in _mode_prox:
                        # si is near
                        if abs(_p-sr) < abs(_p-self.__raw_score_ratio_cum[si - 1]):
                            y = _start_score + si*_step
                            y_pos = si
                        # si-1 is near
                        elif abs(_p-sr) > abs(_p-self.__raw_score_ratio_cum[si - 1]):
                            y = _start_score + (si - 1)*_step
                            y_pos = si - 1
                        # dist is same
                        else:
                            if 'near_max' in _mode_prox:
                                y = _start_score + si*_step
                            else:
                                y = _start_score + (si - 1)*_step
                                y_pos = si - 1
                    else:
                        self.__logger.log('Error Ratio Prox Mode: {}'.format(_mode_prox),
                                             'debug')
                        raise ValueError
                    break
            if y is not None:
                row[col+'_ts'] = slib.round45(y, self.__value_out_score_decimals)
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                section_index_dict.update({(_seg, _seg): y_pos})
                result_ratio.append(_p if row[col+'_count'] > 0 else -1)
            # end scanning raw_score_ratio_list
        # end scanning maptable

        self.__result_formula_coeff = coeff_dict
        formula_dict = {k: '{cf[0][0]:.8f} * (x - {cf[1][1]:.0f}) + {cf[0][1]:.0f}'.format(cf=coeff_dict[k])
                        for k in coeff_dict}
        self.result_dict[col] = {
                                 'coeff': coeff_dict,
                                 'formula': formula_dict
                                 }

        result_section = [coeff_dict[k][1] for k in coeff_dict.keys()]
        result_def_ratio = [self.__raw_score_ratio_cum[section_index_dict[x]]
                            for x in result_section]
        self.__result_ratio_dict[col] = dict()
        self.__result_ratio_dict[col]['def'] = result_def_ratio
        self.__result_ratio_dict[col]['dest'] = result_def_ratio
        self.__result_ratio_dict[col]['match'] = result_ratio
        self.__result_ratio_dict[col]['section'] = result_section
        return True

    def __get_formula_plt(self, field):
        # --step 1
        # claculate raw_score_endpoints
        # self.logger.loginfo('   get raw score section ...')
        if not self.__get_raw_section(field=field):
            return False

        # display ratio searching result at section i
        if self.__logger:
            for i, (cumu_ratio, dest_ratio, match, raw_sec, out_sec) in \
                enumerate(zip(self.__result_ratio_dict[field]['def'],
                              self.__result_ratio_dict[field]['dest'],
                              self.__result_ratio_dict[field]['match'],
                              self.__result_ratio_dict[field]['section'],
                              self.__out_score_section,
                              )):
                self.__logger.log('   <{0:02d}> ratio: [def:{1:.4f}  dest:{2:.4f}  match:{3:.4f}] => '
                      'section_map: raw:[{4:3d}, {5:3d}] --> out:[{6:3d}, {7:3d}]'.
                                  format(i + 1,
                                         cumu_ratio,
                                         dest_ratio,
                                         match,
                                         int(raw_sec[0]),
                                         int(raw_sec[1]),
                                         int(out_sec[0]),
                                         int(out_sec[1])
                                         ),
                                  'debug'
                                  )

        # --step 2
        # calculate Coefficients
        self.__get_formula_coeff(field)
        self.result_dict[field] = {'coeff': copy.deepcopy(self.__result_formula_coeff)}
        return True

    def __get_formula_coeff(self, field=None):
        # coeff: (a, b), (x1, x2), (y1, y2)
        # formula-1: y = (y2-y1)/(x2 -x1)*(x - x1) + y1                   # a(x - b) + c
        #        -2: y = (y2-y1)/(x2 -x1)*x + (y1x2 - y2x1)/(x2 - x1)     # ax + b
        #        -3: y = [(y2-y1)*x + y1x2 - y2x1]/(x2 - x1)              # (ax + b) / c ; int / int

        x_list = self.__result_ratio_dict[field]['section']

        # calculate coefficient
        y_list = self.__out_score_section
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            v = x[1] - x[0]
            if v == 0:
                a = 0
                # mode_section_degraded
                _mode_section_degraded = self.__strategy_dict['mode_section_degraded']
                if _mode_section_degraded == 'to_max':         # x1 == x2 : y = max(y1, y2)
                    b = max(y)
                elif _mode_section_degraded == 'to_min':       # x1 == x2 : y = min(y1, y2)
                    b = min(y)
                elif _mode_section_degraded == 'to_mean':      # x1 == x2 : y = mean(y1, y2)
                    b = np.mean(y)
                else:
                    self.__logger.loginfo('error mode_section_degraded value: {}'.format(_mode_section_degraded))
                    raise ValueError
            else:
                a = (y[1]-y[0])/v                   # (y2 - y1) / (x2 - x1)
                b = (y[0]*x[1]-y[1]*x[0])/v         # (y1x2 - y2x1) / (x2 - x1)
            self.__result_formula_coeff.\
                update({i: [(a, b), (int(x[0]), int(x[1])), (int(y[0]), int(y[1]))]})
        return True

    # new at 2019-09-09
    # extract section points(first end point of first section and second point of all section) from maptable
    #   according ratios in preset ratio_list: raw_score_ratio_cum (cumulative ratio list)
    def __get_raw_section(self, field):
        result_matched_ratio = []
        result_dest_ratio = []
        _ratio_cum_list = self.__raw_score_ratio_cum

        section_real_min = self.df[field].min()
        section_real_max = self.df[field].max()
        section_defined_min = self.__raw_score_defined_min
        section_defined_max = self.__raw_score_defined_max

        _mode_cumu = self.__strategy_dict['mode_section_point_ratio_cumu']
        _mode_order = self.__strategy_dict['mode_score_sort_order']

        # first points of first section in raw score
        if self.__strategy_dict['mode_section_point_first'] == 'real':
            section_start_point = section_real_min \
                if _mode_order in ['a', 'ascending'] else section_real_max
        else:
            section_start_point = section_defined_min \
                if _mode_order in ['a', 'ascending'] else section_defined_max

        result_section_point = [section_start_point]

        # ratio: preset,  percent: computed from data in maptable
        last_ratio = 0
        last_match = 0
        if self.__strategy_dict['mode_section_point_start'] == 'share':
            _step = 0
        else:
            _step = 1 if _mode_order in ['a', 'ascending'] else -1
        for i, cumu_ratio in enumerate(_ratio_cum_list):
            this_seg_ratio = cumu_ratio-last_ratio
            dest_ratio = cumu_ratio if _mode_cumu == 'no' else this_seg_ratio + last_match
            result_dest_ratio.append(dest_ratio)

            # match percent by dest_ratio to get endpoint of this section from maptable
            this_section_end_point, this_match_ratio = \
                self.__get_seg_from_maptable(field, dest_ratio)

            # last section at bottom
            if last_match >= 1:
                this_section_end_point = -1

            # save to result ratio
            result_matched_ratio.append(this_match_ratio)

            # save result endpoints (noshare, share)
            result_section_point.append(this_section_end_point)

            # set last point if mode_section_point_last == 'defined'
            if i > 1:
                # current point: item[i+1] == -1
                if (result_section_point[i+1] < 0) and (result_section_point[i] >= 0):
                    if self.__strategy_dict['mode_section_point_last'] == 'defined':
                        if self.__strategy_dict['mode_score_sort_order'] in ['d', 'descending']:
                            result_section_point[i] = section_defined_min
                        else:
                            result_section_point[i] = section_defined_max

            # save last segment endpoint and percent
            last_ratio = cumu_ratio
            last_match = this_match_ratio

        # step-2: process last same point
        for i in range(len(result_section_point)-1, 1, -1):
            if i == (len(result_section_point)-1):
                if result_section_point[i] == result_section_point[i-1]:
                    result_section_point[i] = -1
            else:
                if result_section_point[i+1] < 0:
                    if result_section_point[i] == result_section_point[i - 1]:
                        result_section_point[i] = -1

        # set last point again if [-1] >= 0 else alread set in loop
        if self.__strategy_dict['mode_section_point_last'] == 'defined':
            for i in range(len(result_section_point)-1, 0, -1):
                if result_section_point[i] >= 0:
                    if self.__strategy_dict['mode_score_sort_order'] in ['d', 'descending']:
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
            _step = -1 if self.__strategy_dict['mode_score_sort_order'] in ['d', 'descending'] else 1
            _y = y
            _x = None
            if x != y:
                if self.__strategy_dict['mode_section_point_start'] == 'share':
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
            if self.__strategy_dict['mode_section_lost'] == 'zip':
                if (_x < 0) or (_y < 0):
                    continue
            make_section.append((_x, _y))
            i += 1
        # add [-1, -1] to tail, fill up that removed in make_section because 'zip'
        len_less = len(self.__raw_score_ratio_cum) - len(make_section)
        if len_less > 0:
            make_section += [[-1, -1] for _ in range(len_less)]

        self.__result_ratio_dict[field] = {
            'def': self.__raw_score_ratio_cum,
            'dest': result_dest_ratio,
            'match': result_matched_ratio,
            'section': make_section
            }

        self.__result_raw_endpoints = result_section_point
        self.raw_section = make_section

        return True

    # new at 2019-09-09
    def __get_seg_from_maptable(self, field, dest_ratio):

        _mode_prox = self.__strategy_dict['mode_section_point_ratio_prox']
        _top_index = self.maptable.index.max()
        _tiny = self.__value_tiny_value

        _seg = -1
        _percent = -1
        last_percent = -1
        last_seg = None
        dist_to_last = 1000
        use_last_seg = False
        for index, row in self.maptable.iterrows():
            _percent = row[field+'_percent']
            _seg = row['seg']
            dist_to_this = abs(_percent - dest_ratio)

            # at bottom, use_current
            if (index == _top_index) or (_percent >= 1):
                break

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
                    self.__logger.loginfo('Error ratio prox mode: {}'.format(_mode_prox))
                    raise ValueError
                break
            dist_to_last = dist_to_this
            last_seg = _seg
            last_percent = _percent

        if use_last_seg:
            return last_seg, last_percent
        return _seg, _percent

    # create report and col_ts in maptable
    def __make_report(self):
        if len(self.__report_doc) == 0:
            self.__report_doc += 'strategies' + '\n'
        else:
            self.__report_doc += '\n strategies' + '\n'
        for k in self.__strategy_dict.keys():
            self.__report_doc += ' ' * 15 + '{0:<30s} {1}'. \
                format(k + ' = ', self.__strategy_dict[k]) + '\n'
        for col in self.cols:
            self.__report_doc += self.__make_field_report(col)
        self.__report_doc += '---' * 40

    def __make_field_report(self, field=''):
        # report start
        if self.model_type == 'plt':
            self.__report_formula_text(field)
        # tiltle
        _out_report_doc = '-'*120 + '\n'
        field_title = '<< score field: [{}] >>\n' + \
                      '- -'*40 + '\n'
        _out_report_doc += field_title.format(field)

        # running result
        if self.model_type == 'plt':
            _out_report_doc += self.__report_running_result_plt(field)
        elif self.model_type == 'ppt':
            _out_report_doc += self.__report_running_result_ppt(field)

        # formula
        if self.model_type == 'plt':
            _out_report_doc += '- -'*40 + '\n'
            _out_report_doc += '< transforming formulas >\n'
            for i, col in enumerate(self.__result_formula_text_list):
                _out_report_doc += ' '*18 + '{}\n'.format(col)

        # statistics
        _out_report_doc += self.__report_statistics(field)

        # shift
        _out_report_doc += '- -'*40 + '\n'
        _diff_raw_out = self.outdf[field+'_ts']-self.outdf[field]
        shift_str = '< score shift>\n' + \
                    'shift_max:'.rjust(17) + ' {:3.1f}\n'.format(max(_diff_raw_out)) + \
                    'shift_min:'.rjust(17) + ' {:3.1f}\n'.format(min(_diff_raw_out)) + \
                    'shift_down(%):'.rjust(17) + ' {:.2f}\n'.\
                        format(_diff_raw_out[_diff_raw_out < 0].count()/_diff_raw_out.count()*100)
        _out_report_doc += shift_str
        _diff_list = []
        for coeff in self.result_dict[field]['coeff'].values():           # self.result_formula_coeff.values():
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
                _diff_list.append((int(rseg[0]), int(slib.round45(b / (1 - a)))))
            if (rseg[0] < oseg[0]) and (rseg[1] >= oseg[1]):
                _diff_list.append((int(slib.round45(b / (1 - a), 0)), int(rseg[1])))

        _out_report_doc += 'shift-down(s):'.rjust(17) + ' ' + str(_diff_list) + ' => '
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
        # _out_report_doc += '---'*40

        # report_doc end
        return _out_report_doc

    def __report_formula_text(self, field):
        p = 0 if self.__strategy_dict['mode_score_sort_order'] in ['ascending', 'a'] else 1
        score_dict = {x: y for x, y in zip(self.maptable['seg'], self.maptable[field+'_count'])}
        self.__result_formula_text_list = []
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
                self.__result_formula_text_list += \
                    ['(section-{0:3d}):  y = {1:0.8f}*(x-{2:2d}) + {3:2d}'.
                     format(int(_fi), formula[0][0], int(formula[1][p]), int(formula[2][p]))]
            else:
                if formula[2][0] != formula[2][1]:
                    self.__result_formula_text_list += \
                        ['(section-{0:3d}):  y = {1:0.8f}*(x-{2:3d}) + {3}({4:3d}, {5:3d})'.
                         format(int(_fi),
                                formula[0][0], formula[1][p],
                                self.__strategy_dict['mode_section_degraded'],
                                formula[2][0], formula[2][1])
                         ]
                # y2 == y1
                else:
                    self.__result_formula_text_list += \
                        ['(section-{0:3d}):  y = 0 * (x-{2:5.2f}) + {3:5.2f}'.
                         format(_fi,
                                formula[0][0],
                                formula[1][p],
                                formula[2][0])
                         ]
            _fi += 1

    def __report_running_result_plt(self, field):
        # def_ratio, dest_ratio, match_ratio, raw_sec, out_sec
        _out_report_doc = ''
        _out_report_doc += '< running result >\n'
        plist = self.__raw_score_ratio_cum
        if self.__value_out_score_decimals == 0:
            _out_report_doc += '  raw score def ratio: [{}]\n'.\
                format(', '.join([format(x, '10.6f') for x in self.__result_ratio_dict[field]['def']]))
            _out_report_doc += '           dest ratio: [{}]\n'.\
                format(', '.join([format(float(x), '10.6f')
                                  for x in self.__result_ratio_dict[field]['dest']]))
            _out_report_doc += '          match ratio: [{}]\n'. \
                format(', '.join([format(float(x), '10.6f') if x > 0 else '***'.rjust(10)
                                  for x in self.__result_ratio_dict[field]['match']]))
        else:
            _out_report_doc += '  raw score sec ratio: [{}]\n'.\
                format(', '.join([format(plist[j]-plist[j-1] if j > 0 else plist[0], '16.6f')
                        for j in range(len(plist))]))
            _out_report_doc += '           cumu ratio: [{}]\n'.\
                format(', '.join([format(x, '16.6f') for x in self.__raw_score_ratio_cum]))
            _out_report_doc += '          match ratio: [{}]\n'.\
                format(', '.join([format(float(x), '16.6f') for x in self.__result_ratio_dict[field]['match']]))

        # get raw segment from result_dict
        _raw_seg_list = [c[1] for c in self.result_dict[field]['coeff'].values()]
        if self.__value_out_score_decimals == 0:
            _out_report_doc += '              section: [{}]\n'.\
                format(', '.join(['({:3d}, {:3d})'.format(int(x), int(y)) for x, y in _raw_seg_list]))
        else:
            _out_report_doc += '              section: [{}]\n'.\
                format(', '.join(['({:6.2f}, {:6.2f})'.format(x, y) for x, y in _raw_seg_list]))

        # get out segment from result_dict[]['coeff']
        _out_seg_list = [x[2] for x in self.result_dict[field]['coeff'].values()]
        if self.__value_out_score_decimals > 0:
            _out_report_doc += '  out  score  section: [{}]\n'.\
                format(', '.join(['({:6.2f}, {:6.2f})'.format(x, y) for x, y in _out_seg_list]))
        else:
            _out_report_doc += '  out  score  section: [{}]\n'.\
                format(', '.join(['({:>3.0f}, {:>3.0f})'.format(x, y) for x, y in _out_seg_list]))
        return _out_report_doc

    def __report_running_result_ppt(self, field):
        # def_ratio, dest_ratio, match_ratio, raw_sec, out_sec
        _raw_seg_list = [c[1] for c in self.result_dict[field]['coeff'].values()]
        _out_seg_list = [x[2] for x in self.result_dict[field]['coeff'].values()]

        column_width = 15
        _out_report_doc = '< running result >\n' + '-'*column_width*5 + '\n'
        _fstr = '{:' + f'{column_width}.8f' + '}'
        _dstr = '{:' + f'{column_width}d' + '}'
        _out_report_doc += 'def_ratio'.rjust(column_width) + 'dest_ratio'.rjust(column_width) + \
                           'match_ratio'.rjust(column_width) + 'raw_score'.rjust(column_width) + \
                           'out_score'.rjust(column_width) + '\n'
        for _def, _dest, _match, _raw, _out in zip(self.__result_ratio_dict[field]['def'],
                                                   self.__result_ratio_dict[field]['dest'],
                                                   self.__result_ratio_dict[field]['match'],
                                                   _raw_seg_list,
                                                   _out_seg_list
                                                   ):
            if _match < 0:
                continue
            _out_report_doc += (_fstr*3 + _dstr*2 + '\n').\
                                format(_def, _dest, _match, int(_raw[0]), int(_out[0]))
        return _out_report_doc

    def __report_formula(self, field):
        pass

    def __report_statistics(self, field):
        _out_report_doc = '- -'*40 + '\n'
        _out_report_doc += '< statistics >\n'

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
        _out_report_doc += ' '*12 + ' raw: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                           format(_max, _min, __mean, _median, _mode)
        _out_report_doc += ' '*18 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                           format(__std, __std/__mean, _max-_min, _skew, _kurt)

        # _count_zero = self.maptable.query(field+'_count==0')['seg'].values
        _count_non_zero = self.maptable.groupby('seg')[[field+'_count']].sum().query(field+'_count>0').index
        _count_zero = [x for x in range(self.__raw_score_defined_min, self.__raw_score_defined_max + 1)
                       if x not in _count_non_zero]
        _out_report_doc += ' '*18 + 'empty_value={}\n' .\
                           format(set_ellipsis_in_digits_sequence(_count_zero))

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
        _out_report_doc += ' '*13 + 'out: max={:6.2f}, min={:5.2f}, mean={:5.2f}, median={:5.2f}, mode={:6.2f}\n' .\
                           format(_max, _min, __mean, _median, _mode)
        _out_report_doc += ' '*18 + 'std={:6.2f},  cv={:5.2f},  ptp={:6.2f},  skew={:5.2f}, kurt={:6.2f}\n' .\
                           format(__std, __std/__mean, _max-_min, _skew, _kurt)

        # _count_zero = self.maptable.query(field+'_count==0')[field+'_ts'].values
        _count_non_zero = self.maptable.groupby(field+'_ts')[[field+'_count']].sum().query(field+'_count>0').index
        if self.model_type == 'ppt': # msin.Models[self.model_name].type == 'plt':
            _count_zero = [x for x in range(int(self.__out_score_real_min), int(self.__out_score_real_max) + 1)
                           if x not in _count_non_zero]
        else:
            _count_zero = ''
        _out_report_doc += ' '*18 + 'empty_value={}\n' .\
                           format(set_ellipsis_in_digits_sequence(_count_zero))
        _out_report_doc += 'size: '.rjust(18) + 'count={}\n' .\
                           format(self.outdf.count()[0])
        return _out_report_doc

    def __report_score_shift(self, field):
        pass

    def report(self):
        print(self.__report_doc)

    def plot(self, mode='model'):
        plot_task = slib.StmPlot(self.cols, self.maptable, self.raw_section, self.__out_score_section)
        plot_task.plot(mode)


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
