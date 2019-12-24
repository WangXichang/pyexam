# -*- utf-8 -*-


# comments to module
"""
    2018.09.24 -- 2018.11
    2019.09.03 --
    designed for new High Test grade score model
    also for shandong interval linear transform

    stm module description stm模块说明：

    [functions] 模块中的函数
       run(name, df, col, ratio_list, grade_max, grade_diff, mode_score_paper_max, mode_score_paper_min,
           out_score_decimal=0, mode_ratio_prox='near', mode_ratio_cumu='yes')
          运行各个模型的调用函数 calling model function
          ---
          参数描述
          name:str := 'shandong', 'shanghai', 'zhejiang', 'beijing', 'tianjin', 'tai', 'hainan2', 'hainan3', 'hainan4'
          调用山东、上海、浙江、北京、天津、广州、海南、...等模型进行分数转换
          --
          name:= 'zscore', 't_score', 'tlinear'
          计算Z分数、T分数、线性转换T分数
          --
          data: input raw score data, type DataFrame of pandas
          输入原始分数数据，类型为DataFrame
          --
          cols:list := raw score fields
          计算转换分数的字段表，列表或元组，元素为字符串
          --
          ratio_list: ratio list including percent value for each interval of grade score
          对原始分数进行等级区间划分的比例表
          --
          grade_max: max value of grade score
          最大等级分数
          --
          grade_diff: differentiao value of grade score
          等级分差值
          --
          raw_score_range: tuple,
          最大原始分数,最小原始分数
          --
          out_score_decimal: decimal digit number
          输出分数小数位数
          --
          mode_ratio_prox: the method to proxmate ratio value of raw score points
                           通过搜索对应比例的确定等级区间分值点的方式
              'upper_min': get score with min value in bigger 小于该比例值的分值中最大的值
              'lower_max': get score with max value in less 大于该比例值的分值中最小的值
              'near_min': get score with min value in near 最接近该比例值的分值中最小的值
              'near_max': get score with max value in near 最接近该比例值的分值中最大的值

          mode_ratio_cumu: 比例累加控制(2019.09.09)
              'yes': 以区间比例累计方式搜索 look up ratio with cumulative ratio
              'no':  以区间比例独立方式搜索 look up ratio with interval ratio individually

          ---
          usage:调用方式
          [1] import pyex_stm as stm
          [2] m = stm.run(name='shandong', df=data, col=['ls'])
          [3] m.report()
          [4] m.output.head()
          [5] m.save_outdf_to_csv

       plot_models_outscore_hist_graph()
          山东、浙江、上海、北京、天津、广东、湖南方案等级转换分数分布直方图
          plot models distribution hist graph including shandong,zhejiang,shanghai,beijing,tianjin

       round45r(v: float, dec=0)
          四舍五入函数, 用于改进round产生的偶数逼近和二进制表示方式产生的四舍五入误差
          function for rounding strictly at some decimal position
          v： 输入浮点数，
          dec：保留小数位数

       get_norm_dist_table(size=400, std=1, mean=0, stdnum=4)
          生成具有指定记录数（size=400）、标准差(std=1)、均值(mean=0)、截止标准差数（最小最大）(stdnum=4)的正态分布表
          create norm data dataframe with assigned scale, mean, standard deviation, std range


    [classes] 模块中的类
       PltScore: 分段线性转换模型, 山东省新高考改革使用 shandong model
       TaiScore: 台湾等级分数模型 Taiwan college entrance test and middle school achievement test model
       Zscore: Z分数转换模型 zscore model
       Tscore: T分数转换模型 t_score model
"""


# built-in import
import copy
import time
import os
import warnings
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
from stm import modelconfig as mcf
from stm import modellib as mapi

warnings.filterwarnings('ignore')


MODELS_NAME_LIST = mcf.Models.keys()


def about():
    print(__doc__)


# Score Transform Model Interface
# Abstract class
class ScoreTransformModel(abc.ABC):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    基于该类的子类（转换分数模型）：
        Z分数非线性模型(Zscore)
        T分数非线性模型(Tscore）
        T分数线性模型（TscoreLinear),
        新高考等级分数转换分数模型（PltScore）（分段线性转换分数）
        param model_name, type==str
        param df: raw score data, type==datafrmae
        param col: fields in df, assign somr subjects score to transform
        param outdf: transform score data, type==dataframe
    """
    def __init__(self, model_name=''):
        self.model_name = model_name
        if model_name in mcf.Models.keys():
            self.model_type = mcf.Models[model_name].type
        else:
            self.model_type = 'other'

        self.df = pd.DataFrame()
        self.cols = []
        self.raw_score_defined_min = None
        self.raw_score_defined_max = None

        self.out_decimal_digits = 0
        self.out_report_doc = ''

        self.map_table = pd.DataFrame()
        self.outdf = pd.DataFrame()

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

    def check_data(self):
        if not isinstance(self.df, pd.DataFrame):
            print('raw data type={} is not dataframe!'.format(type(self.df)))
            return False
        if (type(self.cols) is not list) | (len(self.cols) == 0):
            print('score fields not assigned in cols: {}!'.format(self.cols))
            return False
        for sf in self.cols:
            if sf not in self.df.columns:
                print('score field {} not in raw data columns {}!'.format(sf, list(self.df.columns)))
                return False
        return True

    def check_parameter(self):
        return True

    def run(self):
        if not self.check_data():
            print('check data find error!')
            return False
        if not self.check_parameter():
            print('parameter checking find errors!')
            return False
        return True

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

    def __init__(self, model_name=''):
        # intit df, outdf, model_name, model_type
        super(PltScore, self).__init__(model_name)

        # new properties for shandong model
        self.raw_score_ratio_cum = []
        self.out_score_points = []
        self.out_decimal_digits = 0
        self.out_score_max = None
        self.out_score_min = None
        self.raw_score_real_max = None
        self.raw_score_real_min = None
        self.raw_score_defined_max = None
        self.raw_score_defined_min = None

        # strategy
        self.strategy_dict = {k: mcf.Strategies[k][0]
                              for k in mcf.Strategies}

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
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set outdf
        if type(cols) is str:
            if cols in self.df.columns:
                self.cols = [cols]
                return True
            else:
                print('invalid field name in cols: {}'.format(cols))
        elif type(cols) not in [list, tuple]:
            print('col set fail!\n not a list or tuple!')
            return False
        elif sum([1 if sf in df else 0 for sf in cols]) != len(cols):
            print('field of cols not in rawdf.columns!')
            return False
        else:
            self.cols = cols
            return True

    # plt
    def set_para(self,
                 raw_score_ratio_tuple=None,
                 out_score_seg_tuple=None,
                 raw_score_defined_min=0,
                 raw_score_defined_max=100,
                 mode_ratio_prox='upper_min',
                 mode_ratio_cumu='no',
                 mode_sort_order='descending',
                 mode_section_point_first='real',
                 mode_section_degraded='map_to_max',
                 mode_seg_end_share='no',
                 out_decimal_digits=None):
        if len(raw_score_ratio_tuple) != len(out_score_seg_tuple):
            print('the number of input score points is not same as output score points!')
            return
        if mode_ratio_cumu not in 'yes, no':
            print('mode_ratio_cumu value error:{}'.format(mode_ratio_cumu))

        if isinstance(out_decimal_digits, int):
            self.out_decimal_digits = out_decimal_digits

        if mode_sort_order in ['descending', 'd']:
            raw_p = raw_score_ratio_tuple
            out_pt = out_score_seg_tuple
            self.out_score_points = out_pt
        else:
            raw_p = raw_score_ratio_tuple[::-1]
            out_pt = out_score_seg_tuple[::-1]
            self.out_score_points = tuple(x[::-1] for x in out_pt)
        self.raw_score_ratio_cum = tuple(sum(raw_p[0:x + 1]) for x in range(len(raw_p)))

        self.raw_score_defined_min, self.raw_score_defined_max = raw_score_defined_min, raw_score_defined_max

        self.strategy_dict['mode_ratio_prox'] = mode_ratio_prox
        self.strategy_dict['mode_ratio_cumu'] = mode_ratio_cumu
        self.strategy_dict['mode_sort_order'] = mode_sort_order
        self.strategy_dict['mode_section_point_first'] = mode_section_point_first
        self.strategy_dict['mode_section_degraded'] = mode_section_degraded
        # self.strategy_dict['mode_seg_end_share'] = mode_seg_end_share

    def check_parameter(self):
        if not self.cols:
            print('no score field assign in col!')
            return False
        if (type(self.raw_score_ratio_cum) != tuple) | (type(self.out_score_points) != tuple):
            print('raw_scorepoints or stdscorepoints is not tuple type!')
            return False
        if (len(self.raw_score_ratio_cum) != len(self.out_score_points)) | \
                len(self.raw_score_ratio_cum) == 0:
            print('ratio_tuple len==0 or len(raw_ratio)!=len(out_points)! \nraw={} \nout={}'.
                  format(self.raw_score_ratio_cum, self.out_score_points))
            return False
        return True
    # --------------data and para setting end

    # plt score run
    def run(self):

        print('stm-run begin...\n'+'='*110)
        stime = time.time()

        # check valid
        if not super(PltScore, self).run():
            return

        if self.out_score_points is not None:
            self.out_score_real_max = max([max(x) for x in self.out_score_points])
            self.out_score_real_min = min([min(x) for x in self.out_score_points])

        # calculate seg table
        print('--- calculating map_table ...')
        _segsort = 'a' if self.strategy_dict['mode_sort_order'] in ['ascending', 'a'] else 'd'
        seg_model = mapi.run_seg(
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
        for f in self.cols:
            max_sum = max(self.map_table[f+'_sum'])
            max_sum = 1 if max_sum == 0 else max_sum
            self.map_table[f+'_fr'] = \
                self.map_table[f+'_sum'].apply(lambda x: fr.Fraction(x, max_sum))
            # self.map_table.astype({f+'_fr': fr.Fraction})     # encounter error in python 3.7.4

        # transform score on each field
        self.result_dict = dict()
        self.outdf = self.df.copy(deep=True)
        for i, col in enumerate(self.cols):
            print('--- transform score field:[{}]'.format(col))

            # there is a problem: max set to locale value to each col
            self.raw_score_real_max = self.df[col].max()
            self.raw_score_real_min = self.df[col].min()

            # get formula and save
            _get_formula = False
            if mcf.Models[self.model_name].type == 'ppt':
                _get_formula = self.get_formula_ppt(col)
            else:
                _get_formula = self.get_formula_ts(col)
            if not _get_formula:
                print('getting plt formula fail !')
                return

            # get field_ts in outdf
            print('   calculate: data[{0}] => {0}_ts'.format(col))
            self.outdf.loc[:, (col + '_ts')] = \
                self.outdf[col].apply(
                    lambda x: self.get_ts_score_from_formula_fraction(col, x))

        # create col_ts in map_table
        df_map = self.map_table
        for col in self.cols:
            print('   calculate: map_table[{0}] => [{0}_ts]'.format(col))
            col_name = col + '_ts'
            df_map.loc[:, col_name] = df_map['seg'].apply(
                lambda x: self.get_ts_score_from_formula_fraction(col, x))

        # make report doc
        self.make_report()

        print('='*110)
        print('stm-run end, elapsed-time:', time.time() - stime)

    # run end

    # -----------------------------------------------------------------------------------
    # formula-1
    # y = a*x + b
    # a = (y2-y1)/(x2-x1)
    # b = -x1/(x2-x1) + y1
    def get_ts_score_from_formula_ax_b(self, field, x):
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                return mapi.round45r(cf[0][0] * x + cf[0][1])
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
                    return mapi.round45r(cf[0][1])
                a = (cf[2][1]-cf[2][0])/v
                b = cf[1][0]
                c = cf[2][0]
                return mapi.round45r(a * (x - b) + c)
        return -1

    # -----------------------------------------------------------------------------------
    # formula-3 new, recommend to use,  int/int to float
    # original: y = (y2-y1)/(x2-x1)*(x-x1) + y1
    # variant:  y = (a*x + b) / c
    #           a=(y2-y1)
    #           b=y1x2-y2x1
    #           c=(x2-x1)
    def get_ts_score_from_formula_fraction(self, field, x):
        if x > self.raw_score_defined_max:
            # raise ValueError
            return self.out_score_real_max
        if x < self.raw_score_defined_min:
            # raise ValueError
            return self.out_score_real_min
        for cf in self.result_dict[field]['coeff'].values():
            if cf[1][0] <= x <= cf[1][1] or cf[1][0] >= x >= cf[1][1]:
                a = (cf[2][1]-cf[2][0])
                b = cf[2][0]*cf[1][1] - cf[2][1]*cf[1][0]
                c = (cf[1][1]-cf[1][0])
                if c == 0:  # x1 == x2: use mode_section_degraded: max, min, mean(y1, y2)
                    if self.strategy_dict['mode_section_degraded'] == 'map_to_max':
                        return max(cf[2])
                    elif self.strategy_dict['mode_section_degraded'] == 'map_to_min':
                        return min(cf[2])
                    elif self.strategy_dict['mode_section_degraded'] == 'map_to_mean':
                        return mapi.round45r(np.mean(cf[2]))
                    else:
                        return -1
                return mapi.round45r((a*x + b)/c, self.out_decimal_digits)
        return -1

    # formula hainan, each segment is a single point
    # y = x for x in [x, x]
    # coeff: (a=0, b=x), (x, x), (y, y))
    # len(ratio_list) = len(map_table['seg'])
    def get_formula_ppt(self, col):
        self.result_raw_endpoints = [x for x in self.map_table['seg']]
        self.map_table.loc[:, col+'_ts'] = -1
        coeff_dict = dict()
        result_ratio = []
        _tiny = 10**-8     # used to judge zero(s==0) or equality(s1==s2)

        _mode_sort = self.strategy_dict['mode_sort_order']
        _mode_ppt_score_min = self.strategy_dict['mode_section_point_last']  # real_min or paper_min
        _mode_ppt_score_max = self.strategy_dict['mode_section_point_first']  # real_max or paper_max
        _mode_prox = self.strategy_dict['mode_ratio_prox']

        _start_score = self.out_score_real_max if _mode_sort in ['descending', 'd'] else self.out_score_real_min
        _step = -1 if _mode_sort in ['descending', 'd'] else 1

        # _ts_list = []
        map_table = self.map_table
        real_min = map_table.query(col+'_count>0')['seg'].min()
        real_max = map_table.query(col+'_count>0')['seg'].max()
        for ri, row in map_table.iterrows():
            _seg = row['seg']
            _p = row[col + '_percent']
            y = -1  # init out_score y = a * x + b

            # case: raw score == 0
            # mode_min: , ignore
            # if abs(_seg) < _tiny:
            #     y = self.out_score_real_min
            #     row[col + '_ts'] = y
            #     coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
            #     result_ratio.append(format(_p, '.6f'))
            #     continue
            if _seg == real_min:
                if _mode_ppt_score_min == 'map_to_min':
                    y = self.out_score_real_min
            if _seg == real_max:
                if _mode_ppt_score_max == 'map_to_max':
                    y = self.out_score_real_max
            if y > 0:
                row[col + '_ts'] = y
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                result_ratio.append(format(_p, '.6f'))
                continue

            # loop: seeking ratio by percent to set out score
            for si, sr in enumerate(self.raw_score_ratio_cum):
                # sr == _p or sr > _p
                if (abs(sr - _p) < _tiny) or (sr > _p):
                    if (abs(_p - sr) < _tiny) or (si == 0):
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
                        print('Error Ratio Prox Mode: {}'.format(_mode_prox))
                        raise ValueError
                    break
            if y > 0:
                # print('-1', row[col+'_ts'])
                row[col+'_ts'] = y
                # print('plt', row[col+'_ts'])
                coeff_dict.update({ri: [(0, y), (_seg, _seg), (y, y)]})
                result_ratio.append(format(_p, '.6f'))
                # _ts_list.append(y)
            # end scanning raw_score_ratio_list
        # end scanning map_table

        # if len(self.map_table[col+'_ts']) == len(_ts_list):
        #     map_table.loc[:, col+'_ts'] = _ts_list

        self.result_formula_coeff = coeff_dict
        formula_dict = {k: '{cf[0][0]:.8f} * (x - {cf[1][1]:.0f}) + {cf[0][1]:.0f}'.format(cf=coeff_dict[k])
                        for k in coeff_dict}
        self.result_dict[col] = {
                                 'coeff': coeff_dict,
                                 'formula': formula_dict
                                 }
        self.result_ratio_dict[col] = result_ratio
        return True

    def get_formula_ts(self, field):
        # --step 1
        # claculate raw_score_endpoints
        print('   get input score endpoints ...')
        points_list = self.get_section_first_points_list(field=field)
        self.result_raw_endpoints = points_list
        if len(points_list) == 0:
            return False
        # --step 2
        # calculate Coefficients
        self.get_formula_coeff()
        self.result_dict[field] = {'coeff': copy.deepcopy(self.result_formula_coeff)}
        return True

    # -----------------------------------------------------------------------------------
    # formula-1: y = (y2-y1)/(x2 -x1)*(x - x1) + y1                   # a(x - b) + c
    #        -2:   = (y2-y1)/(x2 -x1)*x + (y1x2 - y2x1)/(x2 - x1)     # ax + b
    #        -3:   = [(y2-y1)*x + y1x2 - y2x1]/(x2 - x1)              # (ax + b) / c ; int / int
    def get_formula_coeff(self):

        # create raw score segments list
        x_points = self.result_raw_endpoints
        step = 1 if self.strategy_dict['mode_sort_order'] in ['ascending', 'a'] else -1
        x_list = [(int(p[0])+(step if i > 0 else 0), int(p[1]))
                  for i, p in enumerate(zip(x_points[:-1], x_points[1:]))]
        # 3-problems: minus score,
        #             less than min,
        #             no ratio interval(not found because of last too large ratio!)
        x_list = [(-1, -1)
                  if p[0] < 0 or min(p) < self.raw_score_defined_min or (p[0]-p[1])*step > 0
                  else p
                  for p in x_list]

        # calculate coefficient
        y_list = self.out_score_points
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            v = x[1] - x[0]
            if v == 0:
                a = 0
                # mode_section_degraded
                _mode_section_degraded = self.strategy_dict['mode_section_degraded']
                if _mode_section_degraded == 'map_to_max':         # x1 == x2 : y = max(y1, y2)
                    b = max(y)
                elif _mode_section_degraded == 'map_to_min':       # x1 == x2 : y = min(y1, y2)
                    b = min(y)
                elif _mode_section_degraded == 'map_to_mean':      # x1 == x2 : y = mean(y1, y2)
                    b = np.mean(y)
                else:
                    print('error mode_section_degraded value: {}'.format(_mode_section_degraded))
                    raise ValueError
            else:
                a = (y[1]-y[0])/v                   # (y2 - y1) / (x2 - x1)
                b = (y[0]*x[1]-y[1]*x[0])/v         # (y1x2 - y2x1) / (x2 - x1)
            self.result_formula_coeff.update({i: [(a, b), x, y]})
        return True

    # new at 2019-09-09
    # extract section points(first end point of first section and second point of all section) from map_table
    #   according ratios in preset ratio_list: raw_score_ratio_cum (cumulative ratio list)
    def get_section_first_points_list(self, field):
        result_ratio = []
        _ratio_cum_list = self.raw_score_ratio_cum

        if self.strategy_dict['mode_section_point_first'] == 'real':
            section_min = self.df[field].min()
            section_max = self.df[field].max()
        else:
            section_min = self.raw_score_defined_min
            section_max = self.raw_score_defined_max

        _mode_cumu = self.strategy_dict['mode_ratio_cumu']
        _mode_order = self.strategy_dict['mode_sort_order']

        # first points of first section in raw score
        section_start_point = section_min if _mode_order in ['a', 'ascending'] else section_max
        result_section_list = [section_start_point]

        # ratio: preset,  percent: computed from data in map_table
        last_ratio = 0
        last_percent = 0
        _step = 1 if _mode_order in ['a', 'ascending'] else -1
        for i, cumu_ratio in enumerate(_ratio_cum_list):
            this_seg_ratio = cumu_ratio-last_ratio
            dest_ratio = cumu_ratio if _mode_cumu == 'no' else this_seg_ratio + last_percent

            # seek endpoint and real cumulative percent of this section from map_table
            this_section_end_point, real_percent = self.get_score_point_from_map_table(field, dest_ratio)

            # first point at first section
            if i == 0:
                this_section_start_point = result_section_list[0]
            # not reached bottom
            elif last_percent <= 1:
                # end_point already in result_seg_list
                if result_section_list[i] != this_section_end_point:
                    # auto get start point by step (1 or -1)
                    this_section_start_point = result_section_list[i] + _step
                else:
                    this_section_start_point = -1
            else:
                # if last endpoint is at bottom, this is set to -1,
                # because of no raw score seg point in this section
                this_section_start_point = -1
                this_section_end_point = -1

            # save to result ratio
            result_ratio.append('{:.6f}'.format(real_percent))
            # save result endpoints (noshare, share)
            result_section_list.append(this_section_end_point)

            # display ratio searching result at section i
            print('   <{0}> ratio: [def:{1:.4f}  real:{2:.4f}  matched:{3:.4f}] => '
                  'section_map: raw:[{4:3.0f}, {5:3.0f}] --> out:[{6:3.0f}, {7:3.0f}]'.
                  format(i+1,
                         cumu_ratio,
                         dest_ratio,
                         real_percent,
                         this_section_start_point,
                         this_section_end_point if this_section_start_point >= 0 else -1,
                         self.out_score_points[i][0],
                         self.out_score_points[i][1]
                         )
                  )

            # save last segment endpoint and percent
            last_ratio = cumu_ratio
            last_percent = real_percent

        self.result_ratio_dict[field] = result_ratio
        return result_section_list

    # new at 2019-09-09
    def get_score_point_from_map_table(self, field, dest_ratio):

        _mode_prox = self.strategy_dict['mode_ratio_prox']
        _top_index = self.map_table.index.max()
        _tiny = 10**-8

        _seg = -1
        _percent = -1
        last_percent = -1
        last_seg = None
        last_diff = 1000
        _use_last = False
        for index, row in self.map_table.iterrows():
            _percent = row[field+'_percent']
            _seg = row['seg']
            _diff = abs(_percent - dest_ratio)

            # at table bottom or lowest score, use_current
            if (index == _top_index) or (_percent >= 1):
                break

            # reach bigger than or equal to ratio
            if _percent >= dest_ratio:
                # at top row
                if last_seg is None:
                    break
                # dealing with strategies
                if 'near' in _mode_prox:
                    # (distances are same, and _mode is near_min) or (last is near)
                    if ((abs(_diff-last_diff) < _tiny) and ('near_min' in _mode_prox)) or \
                       (_diff > last_diff):
                        _use_last = True
                elif _mode_prox == 'lower_max':
                    if abs(_percent-dest_ratio) > _tiny:
                        _use_last = True
                elif _mode_prox == 'upper_min':
                    pass
                else:
                    print('Error ratio prox mode: {}'.format(_mode_prox))
                    raise ValueError
                break
            last_diff = _diff
            last_seg = _seg
            last_percent = _percent
        if _use_last:
            return last_seg, last_percent
        return _seg, _percent

    # create report and col_ts in map_table
    def make_report(self):
        self.out_report_doc = '{}[{}]  {}\n'.\
            format('Transform Model: '.rjust(20),
                   self.model_name,
                   mcf.Models[self.model_name].desc)
        self.out_report_doc += '{}{}\n'.\
            format('running-time: '.rjust(20), time.strftime('%Y.%m.%d  %H:%M:%S', time.localtime()))
        self.out_report_doc += '---'*40 + '\n'
        self.out_report_doc += format('strategies: ', '>20') + '\n'

        for k in mcf.Strategies:
            self.out_report_doc += ' ' * 20 + '{0:<50s} {1}'. \
                format(k + ' = ' + self.strategy_dict[k],
                       mcf.Strategies[k]) + '\n'
        self.out_report_doc += '---'*40 + '\n'
        for col in self.cols:
            print('   create report ...')
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
            for _score in range(formula[1][0], formula[1][1]+_step, _step):
                if score_dict.get(_score, -1) > 0:
                    _break = False
                    break
            if _break:
                continue
            # if formula[1][0] < 0 or formula[1][0] < formula[1][1]:
            #     self.result_formula_text_list += ['(section -{:3d}) ******'.format(_fi)]
            #     continue
            if formula[0][0] > 0:
                self.result_formula_text_list += \
                    ['(section -{0:3d}):  y = {1:0.8f}*(x-{2:2d}) + {3:2d}'.
                     format(_fi, formula[0][0], formula[1][p], formula[2][p])]
            elif formula[0][0] == 0:
                if formula[2][0] != formula[2][1]:
                    self.result_formula_text_list += \
                        ['(section -{0:3d}):  y = {1:0.8f}*(x-{2:3d}) + {3}({4:3d}, {5:3d})'.
                         format(_fi,
                                formula[0][0], formula[1][p],
                                self.strategy_dict['mode_section_degraded'],
                                formula[2][0], formula[2][1])
                         ]
                # y2 == y1
                else:
                    self.result_formula_text_list += \
                        ['(section -{0:.2f}):  y = 0 * (x-{2:.2f}) + {3:.2f}'.
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
        _out_report_doc += '  raw score seg ratio: [{}]\n'.\
            format(', '.join([format(plist[j]-plist[j-1] if j > 0 else plist[0], '0.6f')
                    for j in range(len(plist))]))
        _out_report_doc += '            cum ratio: [{}]\n'.\
            format(', '.join([format(x, '0.6f') for x in self.raw_score_ratio_cum]))
        _out_report_doc += '            get ratio: [{}]\n'.\
            format(', '.join(self.result_ratio_dict[field]))

        # get raw segment from result_dict
        _raw_seg_list = [c[1] for c in self.result_dict[field]['coeff'].values()]
        # if len(_raw_seg_list) > 30:     # for hainan too many segs(801) and single point seg
        #     _raw_seg_list = [x[0] if x[0] == x[1] else x for x in _raw_seg_list]
        _out_report_doc += '            endpoints: [{}]\n'.\
            format(', '.join(['({:3d}, {:3d})'.format(x, y) for x, y in _raw_seg_list]))

        # get out segment from result_dict[]['coeff']
        _out_seg_list = [x[2] for x in self.result_dict[field]['coeff'].values()]
        # if len(_raw_seg_list) > 30:     # for hainan too many segs(801) and single point seg
        #     _out_seg_list = [x[0] if x[0] == x[1] else x for x in _out_seg_list]
        if mcf.Models[self.model_name].type == 'plt':
            _out_report_doc += '  out score endpoints: [{}]\n'.\
                format(', '.join(['({:3d}, {:3d})'.format(x, y) for x, y in _out_seg_list]))
        else:
            _out_report_doc += '  out score endpoints: [{}]\n'.\
                format(', '.join(['({:.2f}, {:.2f})'.format(x, y) for x, y in _out_seg_list]))

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
                           format(mapi.use_ellipsis_in_digits_seq(_count_zero))

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
        if mcf.Models[self.model_name].type == 'plt':
            _count_zero = [x for x in range(self.out_score_real_min, self.out_score_real_max + 1)
                           if x not in _count_non_zero]
        else:
            _count_zero = None
        _out_report_doc += ' '*28 + 'empty_value={}\n' .\
                           format(mapi.use_ellipsis_in_digits_seq(_count_zero))
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
                _diff_list.append((int(rseg[0]), int(mapi.round45r(b/(1-a)))))
            if (rseg[0] < oseg[0]) and (rseg[1] >= oseg[1]):
                _diff_list.append((int(mapi.round45r(b/(1-a), 0)), int(rseg[1])))
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
        if mode not in ['raw', 'out', 'model', 'shift', 'dist', 'diff', 'bar', 'rawbar', 'outbar']:
            print('mode:[{}] is no in [raw, out, model, shift, dist, diff, bar, ourbar, rawbar]'.
                  format(mode))
            return
        if mode in 'shift, model':
            # mode: model describe the differrence of input and output score.
            self.plot_model()
        elif mode in 'dist':
            self._plot_dist_seaborn()
        elif mode in 'bar':
            self.plot_bar('all')
        elif mode == 'rawbar':
            self.plot_bar('raw')
        elif mode == 'outbar':
            self.plot_bar('out')
        elif mode in 'diff':
            self.__plot_diff()
        elif mode in 'normtest':
            self.__plot_norm_test()
        elif not super(PltScore, self).plot(mode):
            print('\"{}\" is invalid'.format(mode))

    def __plot_norm_test(self):
        self.norm_test = dict()
        for col in self.cols:
            _len = self.map_table[col+'_count'].sum()
            x1 = sorted(self.outdf[col])
            x2 = sorted(self.outdf[col+'_ts'])
            y = [(_i-0.375)/(_len+0.25) for _i in range(1, _len+1)]
            fig, ax = plot.subplots()
            # fig.suptitle('norm test for stm models')
            ax.set_title(self.model_name+': norm test')
            ax.plot(x1, y, 'o-', label='score:' + col)
            ax.plot(x2, y, 'o-', label='score:' + col)

    def __plot_diff(self):
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

    def __plot_rawbar(self):
        raw_label = [str(x) for x in range(self.mode_score_paper_max+1)]
        x_data = list(range(self.mode_score_paper_max+1))
        seg_list = list(self.map_table.seg)
        for f in self.cols:
            df = [self.map_table.query('seg=='+str(xv))[f+'_count'].values[0]
                        if xv in seg_list else 0
                        for xv in x_data]
            fig, ax = plot.subplots()
            ax.set_title(self.model_name+'['+f+']: bar graph')
            ax.set_xticks(x_data)
            ax.set_xticklabels(raw_label)
            width = 0.8
            bar_wid = [p - width/2 for p in x_data]
            ax.bar(bar_wid, df, width, label=f)

    def __plot_outbar(self):
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
        raw_label = [str(x) for x in range(self.out_score_real_max + 1)]
        x_data = list(range(self.out_score_real_max + 1))
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
            # bar_wid = [p - width/2 for p in x_data]
            bar_wid = [p + width/2 for p in x_data]
            if display in ['all']:
                raw_bar = ax.bar(bar_wid, df, width, label=f)
                out_bar = ax.bar(bar_wid, outdf, width, label=f + '_ts')
                disp_bar =[raw_bar, out_bar]
            elif 'raw' in display:
                raw_bar = ax.bar(bar_wid, df, width, label=f)
                disp_bar = [raw_bar]
                ax.set_title(self.model_name+'[{}]  mean={:.2f}, std={:.2f}, max={:3d}'.
                             format(f, self.outdf[f].mean(), self.outdf[f].std(), self.outdf[f].max()))
            else:
                out_bar = ax.bar(bar_wid, outdf, width, label=f + '_ts')
                disp_bar = [out_bar]
                ax.set_title(self.model_name + '[{}]  mean={:.2f}, std={:.2f}, max={:3d}'.
                             format(f, self.outdf[f].mean(),
                                    self.outdf[f+'_ts'].std(), self.outdf[f+'_ts'].max()))
            for bars in disp_bar:
                make_color = 0
                for _bar in bars:
                    height = _bar.get_height()
                    height = height - 2 if height > 3 else height
                    xpos = _bar.get_x() + _bar.get_width() / 2
                    # xwid = _bar.get_width()
                    # print(xpos, xwid, height)
                    note_str= '{}'.format(int(height))
                    ax.annotate(note_str,
                                xy=(xpos, height),
                                xytext=(0, 3),              # vertical offset
                                textcoords="offset points",
                                ha='center',
                                va='bottom')
                    if display == 'all':
                        continue
                    if make_color == 2:
                        plot.plot([xpos, xpos], [0, height], hcolor, linewidth=hwidth)
                        make_color = 0
                    else:
                        make_color += 1
            if display == 'all':
                ax.legend(loc='upper right', shadow=True, fontsize='x-large')
            fig.tight_layout()
            plot.show()

    def __plot_dist(self):
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

    def _plot_dist_seaborn(self):
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
            out_min = min([min(p) for p in self.out_score_points])
            out_max = max([max(p) for p in self.out_score_points])

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

    def report_map_table(self):
        fs_list = ['seg']
        for ffs in self.cols:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_ts']
        print(self.map_table[fs_list])


class Zscore(ScoreTransformModel):
    """
    transform raw score to Z-score according to percent position on normal cdf
    input data: df = raw score dataframe
    set parameters: stdNum = standard error numbers
    output data: outdf = result score with raw score field name + '_z'
    """
    # HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self, model_name='zscore'):
        super(Zscore, self).__init__(model_name)

        # input data
        self.df = None
        self.cols = None

        # model parameters
        self.out_score_std_num = 4
        self.raw_score_max = 100
        self.raw_score_min = 0
        self.out_score_decimal = 8
        self.out_score_point_number = 1000
        self.norm_table = array.array('d',
                                      [sts.norm.cdf(-self.out_score_std_num * (1 - 2 * x / (self.out_score_point_number - 1)))
                                       for x in range(self.out_score_point_number)]
                                      )
        # strategies
        self.mode_ratio_prox = 'near'
        self.mode_sort_order = 'd'

        # result data
        self.map_table = None
        self.outdf = None

    def set_data(self, df=None, cols=None):
        self.df = df
        self.cols = cols

    def set_para(self,
                 std_num=4,
                 raw_score_defined_min=0,
                 raw_score_defined_max=100,
                 mode_ratio_prox='near_max',
                 mode_sort_order='d',
                 out_decimal=8,
                 ):
        self.out_score_std_num = std_num
        self.raw_score_max = raw_score_defined_max
        self.raw_score_min = raw_score_defined_min
        self.mode_ratio_prox = mode_ratio_prox
        self.mode_sort_order = mode_sort_order
        self.out_score_decimal = out_decimal

    def check_parameter(self):
        if self.raw_score_max <= self.raw_score_min:
            print('error: max raw score is less than min raw score!')
            return False
        if self.out_score_std_num <= 0:
            print('error: std number {} is error!'.format(self.out_score_std_num))
            return False
        return True

    # Zscore run
    def run(self):
        # check data and parameter in super
        if not super(Zscore, self).run():
            return
        print('start run...')
        st = time.clock()
        self.outdf = self.df
        self.map_table = self.get_map_table(
            self.outdf,
            self.raw_score_max,
            self.raw_score_min,
            self.cols,
            seg_order=self.mode_sort_order)
        for col in self.cols:
            print('calc zscore on field: {}...'.format(col))
            self.map_table[col+'_zscore'] = self.get_zscore(self.map_table[col+'_percent'])
            map_dict = {rscore: zscore for rscore, zscore in
                        zip(self.map_table['seg'], self.map_table[col + '_zscore'])}
            self.outdf.loc[:, col + '_zscore'] = \
                self.outdf[col].apply(lambda x: map_dict.get(x, -999))
        print('zscore finished with {} consumed'.format(round(time.clock()-st, 2)))

    # new method for uniform algorithm with strategies
    def get_zscore(self, percent_list):
        # z_list = [None for _ in percent_list]
        z_array = array.array('d', range(len(percent_list)))
        _len = self.out_score_point_number
        for i, _p in enumerate(percent_list):
            # do not use mode_ratio_prox
            pos = bst.bisect(self.norm_table, _p)
            z_array[i] = 2*(pos - _len/2) / _len * self.out_score_std_num
        return z_array

    @staticmethod
    def get_map_table(df, maxscore, minscore, cols, seg_order='a'):
        seg = mapi.SegTable()
        seg.set_data(df, cols)
        seg.set_para(segmax=maxscore, segmin=minscore, segsort=seg_order)
        seg.run()
        return seg.outdf

    def report(self):
        if type(self.outdf) == pd.DataFrame:
            print('output score desc:\n', self.outdf.describe())
        else:
            print('output score data is not ready!')
        print('data fields in raw_score:{}'.format(self.cols))
        print('para:')
        print('\tzscore stadard diff numbers:{}'.format(self.out_score_std_num))
        print('\tmax score in raw score:{}'.format(self.raw_score_max))
        print('\tmin score in raw score:{}'.format(self.raw_score_min))

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super(Zscore, self).plot(mode)
        else:
            print('not support this mode!')

# === Zscore model end ===


class Tscore(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模,平均数为50,标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出,是为了纪念推孟和桑代克
    对智力测验,尤其是提出智商这一概念所作出的巨大贡献。
    通过调整t_score_mean, t_score_std, 也可以进行其它标准分数转换，
    如100-900分的标准分数转换。
    本模型使用百分位-累计分布校准的方式计算转换分数。
    '''

    def __init__(self, model_name='tscore'):
        super(Tscore, self).__init__(model_name)

        self.mode_score_paper_max = 100
        self.mode_score_paper_min = 0
        self.t_score_std = 10
        self.t_score_mean = 50
        self.t_score_stdnum = 4

        self.outdf_decimal = 0
        self.zscore_decimal = 8

        self.map_table = None

    def set_data(self, df=None, cols=None):
        self.df = df
        self.cols = cols

    def set_para(self, 
                 mode_score_paper_min=0,
                 mode_score_paper_max=100,
                 t_score_mean=50,
                 t_score_std=10,
                 t_score_stdnum=4,
                 out_decimal=0):
        self.mode_score_paper_max = mode_score_paper_max
        self.mode_score_paper_min = mode_score_paper_min
        self.t_score_mean = t_score_mean
        self.t_score_std = t_score_std
        self.t_score_stdnum = t_score_stdnum
        self.outdf_decimal = out_decimal

    # Tscore
    def run(self):
        """get tscore from zscore"""
        zm = Zscore()
        zm.set_data(self.df, self.cols)
        zm.set_para(std_num=self.t_score_stdnum,
                    raw_score_range=(self.mode_score_paper_min, self.mode_score_paper_max),
                    out_decimal=self.zscore_decimal
                    )
        zm.run()
        self.outdf = zm.outdf
        namelist = self.outdf.columns

        def formula(x):
            return mapi.round45r(x * self.t_score_std + self.t_score_mean, self.outdf_decimal)

        for sf in namelist:
            if '_zscore' in sf:
                new_sf = sf.replace('_zscore', '_tscore')
                self.outdf.loc[:, new_sf] = self.outdf[sf].apply(formula)
        self.map_table = zm.map_table

    def report(self):
        print('T-score by normal table transform report')
        print('-' * 50)
        if type(self.df) == pd.DataFrame:
            print('raw score desc:')
            print('    fields:', self.cols)
            print(self.df[self.cols].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.outdf) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.cols]
            print('T-score desc:')
            print('    fields:', out_fields)
            print(self.outdf[out_fields].describe())
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in raw_score:{}'.format(self.cols))
        print('-' * 50)
        print('para:')
        print('\tzscore stadard deviation numbers:{}'.format(self.t_score_std))
        print('\tmax score in raw score:{}'.format(self.mode_score_paper_max))
        print('\tmin score in raw score:{}'.format(self.mode_score_paper_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super(Tscore, self).plot(mode)


class TaiScore(ScoreTransformModel):
    """
    Grade Score Model used by Taiwan College Admission Test Center
    top_group = df.sort_values(field,ascending=False).head(int(df.count(0)[field]*0.01))[[field]]
    high_grade_score = round(top_group[field].mean(), 4)
    intervals = [minscore, grade_level/grade_level_total_number], ..., [,high_grade]
    以原始分值切分，形成的分值相当于等距合并，粒度直接增加
    实质上失去了等级分数的意义
    本模型仍然存在高分区过度合并问题
    """

    def __init__(self, model_name='tai'):
        super(TaiScore, self).__init__(model_name)
        self.model_name = 'Taiwan'

        self.grade_num = 15
        self.mode_score_paper_max = 100
        self.mode_score_paper_min = 0
        self.max_ratio = 0.01   # 1%
        self.df = pd.DataFrame()

        self.grade_no = [x for x in range(self.grade_num+1)]
        self.map_table = None
        self.grade_dist_dict = {}  # col: grade_list, from max to min
        self.outdf = pd.DataFrame()

    def set_data(self, df=pd.DataFrame(), cols=None):
        if len(df) > 0:
            self.df = df
        if isinstance(cols, list) or isinstance(cols, tuple):
            self.cols = cols

    def set_para(self,
                 mode_score_paper_max=None,
                 mode_score_paper_min=None,
                 grade_num=None,
                 ):
        if isinstance(mode_score_paper_max, int):
            if len(self.cols) > 0:
                if mode_score_paper_max >= max([max(self.df[f]) for f in self.cols]):
                    self.mode_score_paper_max = mode_score_paper_max
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set col first!')
        if isinstance(mode_score_paper_min, int):
            self.mode_score_paper_min = mode_score_paper_min
        if isinstance(grade_num, int):
            self.grade_num = grade_num
        self.grade_no = [x for x in range(self.grade_num+1)]

    def run(self):
        self.run_create_grade_dist_list()
        self.run_create_outdf()

    def run_create_grade_dist_list(self):
        # mode_ratio_prox = 'near'
        seg = mapi.SegTable()
        seg.set_para(segmax=self.mode_score_paper_max,
                     segmin=self.mode_score_paper_min,
                     segsort='d')
        seg.set_data(self.df,
                     self.cols)
        seg.run()
        self.map_table = seg.outdf
        for fs in self.cols:
            lastpercent = 0
            lastseg = self.mode_score_paper_max
            for ind, row in self.map_table.iterrows():
                curpercent = row[fs + '_percent']
                curseg = row['seg']
                if row[fs+'_percent'] > self.max_ratio:
                    if curpercent - self.max_ratio > self.max_ratio - lastpercent:
                        max_score = lastseg
                    else:
                        max_score = curseg
                    max_point = self.df[self.df[fs] >= max_score][fs].mean()
                    # print(fs, max_score, curseg, lastseg)
                    self.grade_dist_dict.update({fs: mapi.round45r(max_point/self.grade_num, 8)})
                    break
                lastpercent = curpercent
                lastseg = curseg

    def run_create_outdf(self):
        dt = copy.deepcopy(self.df[self.cols])
        for fs in self.cols:
            dt.loc[:, fs+'_grade'] = dt[fs].apply(lambda x: self.run__get_grade_score(fs, x))
            dt2 = self.map_table
            dt2.loc[:, fs+'_grade'] = dt2['seg'].apply(lambda x: self.run__get_grade_score(fs, x))
            self.outdf = dt

    def run__get_grade_score(self, fs, x):
        if x == 0:
            return x
        grade_dist = self.grade_dist_dict[fs]
        for i in range(self.grade_num):
            minx = i * grade_dist
            maxx = (i+1) * grade_dist if i < self.grade_num-1 else self.mode_score_paper_max
            if minx < x <= maxx:
                return i+1
        return -1

    def plot(self, mode='raw'):
        pass

    def report(self):
        print(self.outdf[[f+'_grade' for f in self.cols]].describe())

    def print_map_table(self):
        print(self.map_table)
