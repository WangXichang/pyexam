# -*- utf-8 -*-

# version 2018-09-24
# revised for discontinuous points
# separate from pyex_lib, pyex_seg
# only rely on pyex_ptt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import time
import scipy.stats as sts
import seaborn as sbn
import pyex_ptt as ptt


# some constants for models
shandong_ratio = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]
shandong_interval = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
shanghai_ratio = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
beijing_ratio = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
tianjin_ratio = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]


def help_doc():
    print("""
    module function and class:
    
    function
       stmmodel: 调用有关模型的接口函数，
          通过指定name=‘shandong'/'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tao'
          可以计算山东、上海、浙江、北京、天津、陶百强模型的转换分数
          也可以计算Z分数、T分数和线性转换T分数（name='zscore'/'tscore'/'tlinear'）
       plot_model_distribution: 
          各方案按照比例转换后分数后的分布直方图
       round45i: 四舍五入函数
       exp_norm_table: 根据均值和标准差数生成正态分布表
    
    class 
       PltScore: 分段线性转换模型, 山东省新高考改革使用
       LevelScore: 等级分数转换模型, 浙江、上海、天津、北京使用
       Zscore: Z分数转换模型
       Tscore: T分数转换模型
       SegTable: 计算分段表模型
    """)


# interface to use model for some typical application
def stmmodel(name='shandong',
             df=None,
             field_list='',
             maxscore=100,
             minscore=0,
             decimal=0,
             approx_method='near'
             ):
    """
    :param name: str, 'shandong', 'shanghai', 'shandong', 'beijing', 'tianjin', 'zscore', 'tscore', 'tlinear'
    :param df: input dataframe
    :param field_list: score fields list in input dataframe
    :param maxscore: max value in raw score
    :param minscore: min value in raw score
    :param decimal: output score decimal digits
    :param approx_method: maxmin, minmax, nearmin, nearmax
    :return: model
    """
    # constant data
    # valid name
    name_set = 'zhejiang, shanghai, shandong, beijing, tianjin, tao, ' \
               'tscore, zscore, tlinear'

    if type(df) != pd.DataFrame:
        if type(df) == pd.Series:
            input_data = pd.DataFrame(df)
        else:
            print('no score dataframe!')
            return
    else:
        input_data = df
    if isinstance(field_list, str):
        field_list = [field_list]
    elif not isinstance(field_list, list):
        print('invalid field_list!')
        return

    if name not in name_set:
        print('invalid name, not in {}'.format(name_set))
        return

    # shandong new project score model
    if name == 'shandong':
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
        #     frequecy estimation: 0.0234 * total_number
        #     200,000-->4680,   300,000 --> 7020

        pltmodel = PltScore()
        pltmodel.model_name = 'shandong'
        pltmodel.output_data_decimal = 0
        pltmodel.set_data(input_data=input_data,
                          field_list=field_list)
        pltmodel.set_parameters(input_score_percent_list=shandong_ratio,
                                output_score_points_list=shandong_interval,
                                input_score_max=maxscore,
                                input_score_min=minscore,
                                approx_mode=approx_method,
                                use_minscore_as_start_endpoint=True,
                                decimals=decimal
                                )
        pltmodel.run()
        return pltmodel

    if name == 'zhejiang':
        # estimate: std = 14, mean=70
        level_score_table = [100 - x * 3 for x in range(21)]
        m = LevelScore()
        m.model_name = 'zhejiang'
        m.set_data(input_data=input_data, field_list=field_list)
        m.set_parameters(maxscore=maxscore,
                         minscore=minscore,
                         level_ratio_table=zhejiang_ratio,
                         level_score_table=level_score_table,
                         approx_method=approx_method
                         )
        m.run()
        return m

    if name == 'shanghai':
        level_score = [70 - j * 3 for j in range(11)]
        m = LevelScore()
        m.model_name = 'shanghai'
        m.set_data(input_data=input_data, field_list=field_list)
        m.set_parameters(level_ratio_table=shanghai_ratio,
                         level_score_table=level_score,
                         maxscore=maxscore,
                         minscore=minscore)
        m.run()
        return m

    if name == 'beijing':
        level_score = [100 - j * 3 for j in range(21)]
        m = LevelScore()
        m.model_name = 'beijing'
        m.set_data(input_data=input_data, field_list=field_list)
        m.set_parameters(level_ratio_table=beijing_ratio,
                         level_score_table=level_score,
                         maxscore=maxscore,
                         minscore=minscore)
        m.run()
        return m

    if name == 'tianjin':
        level_score = [100 - j * 3 for j in range(21)]
        m = LevelScore()
        m.model_name = 'tianjin'
        m.set_data(input_data=input_data, field_list=field_list)
        m.set_parameters(level_ratio_table=tianjin_ratio,
                         level_score_table=level_score,
                         maxscore=maxscore,
                         minscore=minscore)
        m.run()
        return m

    if name == 'tao':
        m = LevelScoreTao()
        m.level_num = 50
        m.set_data(input_data=input_data,
                   field_list=field_list)
        m.set_parameters(maxscore=maxscore,
                         minscore=minscore)
        m.run()
        return m

    if name == 'zscore':
        zm = Zscore()
        zm.model_name = name
        zm.set_data(input_data=input_data, field_list=field_list)
        zm.set_parameters(std_num=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm
    if name == 'tscore':
        tm = Tscore()
        tm.model_name = name
        tm.set_data(input_data=input_data, field_list=field_list)
        tm.set_parameters(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'tlinear':
        tm = TscoreLinear()
        tm.model_name = name
        tm.set_data(input_data=input_data, field_list=field_list)
        tm.set_parameters(input_score_max=100, input_score_min=0)
        tm.run()
        tm.report()
        return tm


def plot_model_distribution():
    plt.figure('model ratio distribution')
    plt.rcParams.update({'font.size': 16})
    plt.subplot(231)
    plt.bar(range(8), [shandong_ratio[j] - shandong_ratio[j - 1] for j in range(1, 9)])
    plt.title('shandong model')

    plt.subplot(232)
    plt.bar(range(11, 0, -1), [shanghai_ratio[-j - 1] for j in range(11)])
    plt.title('shanghai model')

    plt.subplot(233)
    plt.bar(range(21, 0, -1), [zhejiang_ratio[-j - 1] for j in range(len(zhejiang_ratio))])
    plt.title('zhejiang model')

    plt.subplot(234)
    plt.bar(range(21, 0, -1), [beijing_ratio[-j - 1] for j in range(len(beijing_ratio))])
    plt.title('beijing model')

    plt.subplot(235)
    plt.bar(range(21, 0, -1), [tianjin_ratio[-j - 1] for j in range(len(tianjin_ratio))])
    plt.title('tianjin model')


# Score Transform Model Interface
# Abstract class
class ScoreTransformModel(object):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    基于该类的子类（转换分数模型）：
        Z分数非线性模型(Zscore)
        T分数非线性模型(Tscore）
        T分数线性模型（TscoreLinear),
        等级分数模型(LevelScore)
        山东省新高考转换分数模型（PltScore）（分段线性转换分数）
        param model_name, type==str
        param input_data: raw score data, type==datafrmae
        param field_list: fields in input_data, assign somr subjects score to transform
        param output_data: transform score data, type==dataframe
    """
    def __init__(self, model_name=''):
        self.model_name = model_name

        self.input_data = pd.DataFrame()
        self.field_list = []
        self.input_score_min = 0
        self.input_score_max = 100

        self.output_data = pd.DataFrame()
        self.output_data_decimal = 0
        self.output_report_doc = ''

        self.sys_pricision_decimals = 6

    def set_data(self, input_data=None, field_list=None):
        raise NotImplementedError()

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if not isinstance(self.input_data, pd.DataFrame):
            print('rawdf is not dataframe!')
            return False
        if (type(self.field_list) != list) | (len(self.field_list) == 0):
            print('no score fields assigned!')
            return False
        for sf in self.field_list:
            if sf not in self.input_data.columns:
                print('error score field {} !'.format(sf))
                return False
        return True

    def check_parameter(self):
        return True

    def run(self):
        if not self.check_data():
            print('check data find error!')
            return False
        if not self.check_parameter():
            print('check parameter find error!')
            return False
        return True

    def report(self):
        raise NotImplementedError()

    def plot(self, mode='raw'):
        # implemented plot_out, plot_raw score figure
        if mode.lower() == 'out':
            self.__plot_out_score()
        elif mode.lower() == 'raw':
            self.__plot_raw_score()
        else:
            print('error mode={}, valid mode: out or raw'.format(mode))
            return False
        return True

    def __plot_out_score(self):
        if not self.field_list:
            print('no field:{0} assign in {1}!'.format(self.field_list, self.input_data))
            return
        # plt.figure(self.model_name + ' out score figure')
        labelstr = 'Output Score '
        for fs in self.field_list:
            plt.figure(fs)
            if fs+'_plt' in self.output_data.columns:  # find sf_outscore field
                sbn.distplot(self.output_data[fs+'_plt'])
                plt.title(labelstr+fs)
        return

    def __plot_raw_score(self):
        if not self.field_list:
            print('no field assign in rawdf!')
            return
        labelstr = 'Raw Score '
        for sf in self.field_list:
            plt.figure(sf)
            sbn.distplot(self.input_data[sf])
            plt.title(labelstr + sf)
        return


# piecewise linear transform model
class PltScore(ScoreTransformModel):
    """
    PltModel:
    linear transform from raw-score to level-score at each intervals divided by preset ratios
    set ratio and intervals according to norm distribution property
    get a near normal distribution

    # for ratio = [3, 7, 16, 24, 24, 16, 7, 3] & level = [20, 30, ..., 100]
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
    """

    def __init__(self):
        # intit input_df, input_output_data, output_df, model_name
        super(PltScore, self).__init__('plt')

        # new properties for linear segment stdscore
        self.input_score_percentage_points = []
        self.output_score_points = []
        self.output_data_decimal = 0

        # parameters
        self.approx_mode = 'minmax'
        self.use_minscore_as_rawscore_start_endpoint = True

        # result
        self.segtable = pd.DataFrame()
        self.result_input_data_points = []
        self.result_coeff = {}
        self.result_formula = ''
        self.result_dict = {}

    def set_data(self, input_data=None, field_list=None):

        # check and set rawdf
        if type(input_data) == pd.Series:
            self.input_data = pd.DataFrame(input_data)
        elif type(input_data) == pd.DataFrame:
            self.input_data = input_data
        else:
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set output_data
        if not field_list:
            self.field_list = [s for s in input_data]
        elif type(field_list) != list:
            print('field_list set fail!\n not a list!')
            return
        elif sum([1 if sf in input_data else 0 for sf in field_list]) != len(field_list):
            print('field_list set fail!\n field must in rawdf.columns!')
            return
        else:
            self.field_list = field_list

    def set_parameters(self,
                       input_score_percent_list=None,
                       output_score_points_list=None,
                       input_score_min=0,
                       input_score_max=150,
                       approx_mode='minmax',
                       use_minscore_as_start_endpoint=True,
                       decimals=None):
        """
        :param input_score_percent_list: ratio points for raw score interval
        :param output_score_points_list: score points for output score interval
        :param input_score_min: min value to transform
        :param input_score_max: max value to transform
        :param approx_mode:  minmax, maxmin, nearmin, nearmax
        :param use_minscore_as_start_endpoint: set low endpoint for raw score, default None for using real minvalue
        :param decimals: decimal digit number to remain in result data
        """
        if (type(input_score_percent_list) != list) | (type(output_score_points_list) != list):
            print('input score points or output score points is not list!')
            return
        if len(input_score_percent_list) != len(output_score_points_list)+1:
            print('the number of input score points is not same as output score points!')
            return
        if isinstance(decimals, int):
            self.output_data_decimal = decimals
        self.input_score_percentage_points = input_score_percent_list
        self.output_score_points = output_score_points_list
        if isinstance(input_score_min, int):
            self.input_score_min = input_score_min
        if isinstance(input_score_max, int):
            self.input_score_max = input_score_max
        self.approx_mode = approx_mode
        self.use_minscore_as_rawscore_start_endpoint = use_minscore_as_start_endpoint

    def check_parameter(self):
        if not self.field_list:
            print('no score field assign in field_list!')
            return False
        if (type(self.input_score_percentage_points) != list) | (type(self.output_score_points) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return False
        if (len(self.input_score_percentage_points) != len(self.output_score_points)) | \
                len(self.input_score_percentage_points) == 0:
            print('len is 0 or not same for raw score percent and std score points list!')
            return False
        return True

    # --------------data and parameters setting end

    def run(self):
        stime = time.time()

        # check valid
        if not super().run():
            return

        # calculate seg table
        print('--- start calculating segtable ---')
        # import pyex_seg as psg
        seg = SegTable()
        seg.set_data(input_data=self.input_data, field_list=self.field_list)
        seg.set_parameters(segmax=self.input_score_max,
                           segmin=self.input_score_min,
                           segsort='a',
                           segstep=1,
                           display=False)
        seg.run()
        self.segtable = seg.output_data

        # transform score on each field
        self.result_dict = {}
        result_dataframe = None
        result_report_save = ''
        for i, fs in enumerate(self.field_list):
            print(' --start transform score field: <<{}>>'.format(fs))
            # create output_data by filter from df
            _filter = '(df.{0}>={1}) & (df.{2}<={3})'.\
                      format(fs, self.input_score_min, fs, self.input_score_max)
            print('   filter created: [{}]'.format(_filter))
            df = self.input_data
            # df2 = df[eval(_filter)][[fs]]
            self.output_data = df[eval(_filter)][[fs]]

            # get fomula
            if not self.__get_formula(fs):
                print('fail to initializing !')
                return

            # transform score
            print('   begin calculating ...')
            df2 = self.output_data
            score_list = df2[fs].apply(self.__get_plt_score, self.output_data_decimal)
            self.output_data.loc[:, (fs + '_plt')] = score_list
            self.__get_report_doc(fs)
            print('   merge dataframe ...')
            if i == 0:
                result_dataframe = self.input_data.merge(self.output_data[[fs+'_plt']],
                                                         how='left', right_index=True, left_index=True)
            else:
                result_dataframe = result_dataframe.merge(self.output_data[[fs+'_plt']],
                                                          how='left', right_index=True, left_index=True)
            print('   create report ...')
            result_report_save += self.output_report_doc

            # save result
            self.result_dict[fs] = {
                'input_score_points': copy.deepcopy(self.result_input_data_points),
                'coeff': copy.deepcopy(self.result_coeff),
                'formulas': copy.deepcopy(self.result_formula)}

        self.output_report_doc = result_report_save
        self.output_data = result_dataframe.fillna(-1)

        print('used time:', time.time() - stime)
        print('-'*50)
        # run end

    # from current formula in result_coeff
    def __get_plt_score(self, x):
        for f in self.result_coeff.values():
            # print(f)
            if f[1][0] <= x <= f[1][1]:
                return round45i(f[0]*(x - f[1][0]) + f[2][0], self.output_data_decimal)
        return -1

    def __get_formula(self, field):

        # step-prepare
        # check format
        if type(self.input_data) != pd.DataFrame:
            print('no dataset given!')
            return False
        if not self.output_score_points:
            print('no standard score interval points given!')
            return False
        if not self.input_score_percentage_points:
            print('no score interval percent given!')
            return False
        if len(self.input_score_percentage_points) != len(self.output_score_points)+1:
            print('score interval for rawscore and stdscore is not same!')
            print(self.output_score_points, self.input_score_percentage_points)
            return False
        if self.output_score_points != sorted(self.output_score_points):
            print('stdscore points is not in order!')
            return False
        if sum([0 if (x <= 1) & (x >= 0) else 1 for x in self.input_score_percentage_points]) > 0:
            print('raw score interval percent is not percent value !\n', self.input_score_percentage_points)
            return False

        # step 1
        # claculate _rawScorePoints
        if field in self.output_data.columns.values:
            print('-- get input score endpoints ...')
            self.result_input_data_points = self.__get_raw_score_points(field, self.approx_mode)
        else:
            print('score field({}) not in output_dataframe columns:{}!'.format(field, self.output_data.columns.values))
            print('the field should be in input_dataframe columns:{}'.format(self.input_data.columns.values))
            return False

        # step 2
        # calculate Coefficients
        if not self.__get_coeff():
            return False

        return True

    def __get_raw_score_points(self, field, mode='minmax'):
        if mode not in 'minmax, maxmin, nearmax, nearmin':
            print('error mode {} !'.format(mode))
            raise TypeError

        score_points = [self.input_score_min]
        lastpercent = 0
        lastseg = self.input_score_min
        percent_loc = 1
        for index, row in self.segtable.iterrows():
            p = row[field+'_percent']
            thisseg = row['seg']
            cur_input_score_ratio = self.input_score_percentage_points[percent_loc]
            if (p == 1) | (percent_loc == len(self.input_score_percentage_points)):
                score_points += [thisseg]
                break
            if mode in 'minmax, maxmin':
                if p == cur_input_score_ratio:
                    if (row['seg'] == 0) & (mode == 'minmax') & (index < self.input_score_max):
                        pass
                    else:
                        score_points.append(thisseg)
                        percent_loc += 1
                elif p > cur_input_score_ratio:
                    score_points.append(lastseg if mode == 'minmax' else thisseg)
                    percent_loc += 1
            if mode in 'nearmax, nearmin, near':
                if p > cur_input_score_ratio:
                    if (p - cur_input_score_ratio) < abs(cur_input_score_ratio - lastpercent):
                        # thispercent is near to p
                        score_points.append(thisseg)
                    elif (p-cur_input_score_ratio) > abs(cur_input_score_ratio - lastpercent):
                        # lastpercent is near to p
                        score_points.append(lastseg)
                    else:
                        # two dist is equal, to set nearmin if near
                        if mode == 'nearmax':
                            score_points.append(thisseg)
                        else:
                            score_points.append(lastseg)
                    percent_loc += 1
                elif p == cur_input_score_ratio:
                    # some percent is same as input_ratio
                    nextpercent = -1
                    if thisseg < self.input_score_max:  # max(self.segtable.seg):
                        nextpercent = self.segtable['seg'].loc[thisseg + 1]
                    if p == nextpercent:
                        continue
                    # next is not same
                    if p == lastpercent:
                        # two percent is 0
                        if mode == 'nearmax':
                            score_points += [thisseg]
                        else:  # nearmin
                            score_points += [lastseg]
                    else:
                        score_points += [thisseg]
                    percent_loc += 1
            lastseg = thisseg
            lastpercent = p
        return score_points

    # formula: y = (y2-y1-1)/(x2 -x1) * (x - x1) + y1
    # coeff = (y2-y1-1)/(x2 -x1)
    def __get_coeff(self):
        # check length of input_points and output_points
        if len(self.result_input_data_points) != len(self.output_score_points)+1:
            print('error score points: {}'.format(self.result_input_data_points))
            return False
        # calculate coeff
        x_points = self.result_input_data_points
        x_min = min(x_points)
        xp = [(int(x1)+1 if x1 > x_min else x1, int(x2))
              for x1, x2 in zip(x_points[:-1], x_points[1:])]
        yp = self.output_score_points
        # yp = [(int(y1), int(y2)-1 if y2 < y_max else int(y2))
        #       for y1, y2 in zip(y_points[:-1], y_points[1:])]
        for i, p in enumerate(zip(xp, yp)):
            coeff = (p[1][1] - p[1][0]) / (p[0][1] - p[0][0])
            self.result_coeff.update({i: [coeff, p[0], p[1]]})
        return True

    def __get_report_doc(self, field=''):
        self.result_formula = ['{0}*(x-{1})+{2}'.format(round45i(x[0], 16), x[1][0], x[2][0])
                               for x in self.result_coeff.values()]
        self.output_report_doc = '---<< score field: [{}] >>---\n'.format(field)
        self.output_report_doc += 'input score percentage: {}\n'.\
            format(self.input_score_percentage_points)
        self.output_report_doc += 'input score  endpoints: {}\n'.\
            format([x[1] for x in self.result_coeff.values()])
        self.output_report_doc += 'output score endpoints: {}\n'.\
            format([x[2] for x in self.result_coeff.values()])
        for i, fs in enumerate(self.result_formula):
            if i == 0:
                self.output_report_doc += '    transform formulas: {}\n'.format(fs)
            else:
                self.output_report_doc += '                        {}\n'.format(fs)
        self.output_report_doc += '---'*30 + '\n\n'

    # calculate single field from raw score to plt_score
    def rerun_at_field(self, rawscore_field):
        if rawscore_field not in self.field_list:
            print('field:{} not in field_list:{}'.format(rawscore_field, self.field_list))
            return
        # recreate formula
        if not self.__get_formula(rawscore_field):
            print('create formula fail!')
            return
        # transform score
        if not isinstance(self.output_data, pd.DataFrame):
            self.output_data = self.input_data.copy(deep=True)
        self.output_data.loc[:, rawscore_field + '_plt'] = \
            self.input_data[rawscore_field].apply(self.__get_plt_score)
        # create report
        self.__get_report_doc()

    # get score from formula
    def get_plt_score_from_formula(self, field, x):
        if field not in self.field_list:
            print('invalid field name {} not in {}'.format(field, self.field_list))
        coeff = self.result_dict[field]['coeff']
        for cf in coeff.values():
            if cf[1][0] <= x <= cf[1][1]:
                return round45i(cf[0]*(x - cf[1][0]) + cf[2][0],
                                self.output_data_decimal)

    def report(self):
        print(self.output_report_doc)

    def plot(self, mode='model'):
        if mode not in ['raw', 'out', 'model', 'shift']:
            print('valid mode is: raw, out, model,shift')
            # print('mode:model describe the differrence of input and output score.')
            return
        if mode == 'model':
            self.__plotmodel()
        elif not super().plot(mode):
            print('mode {} is invalid'.format(mode))

    def __plotmodel(self):
        # 分段线性转换模型
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams.update({'font.size': 8})
        for i, fs in enumerate(self.field_list):
            result = self.result_dict[fs]
            input_points = result['input_score_points']
            in_max = max(input_points)
            ou_min = min([min(p) for p in self.output_score_points])
            ou_max = max([max(p) for p in self.output_score_points])

            plt.figure(fs+'_plt')
            plt.rcParams.update({'font.size': 10})
            plt.title(u'转换模型({})'.format(fs))
            plt.xlim(min(input_points), max(input_points))
            plt.ylim(ou_min, ou_max)
            plt.xlabel(u'\n原始分数')
            plt.ylabel(u'转换分数')
            plt.xticks([])
            plt.yticks([])

            formula = self.result_dict[fs]['coeff']
            for cf in formula.values():
                x = cf[1]
                y = cf[2]
                plt.plot(x, y)
                for j in [0, 1]:
                    plt.plot([x[j], x[j]], [0, y[j]], '--')
                    plt.plot([0, x[j]], [y[j], y[j]], '--')
                for j, xx in enumerate(x):
                    plt.text(xx-1 if j == 1 else xx, ou_min-2, '{}'.format(int(xx)))
                for j, yy in enumerate(y):
                    plt.text(1, yy-2 if j == 1 else yy, '{}'.format(int(yy)))
            plt.plot((0, in_max), (0, in_max))

        plt.show()
        return

    def print_segtable(self):
        seg_decimal_digit = 8
        fs_list = ['seg']
        for ffs in self.field_list:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_plt']
        df = self.segtable
        for fs in fs_list:
            if 'percent' in fs:
                df[fs] = df[fs].apply(lambda x: round(x, seg_decimal_digit))
            if '_plt' in fs:
                df.loc[:, fs] = df['seg'].apply(
                    lambda x: self.get_plt_score_from_formula(fs[0:fs.index('_')], x))
        print(ptt.make_page(df=df[fs_list],
                            title='level score table for {}'.format(self.model_name),
                            pagelines=self.input_score_max+1))


class Zscore(ScoreTransformModel):
    """
    transform raw score to Z-score according to percent position on normal cdf
    input data: 
    rawdf = raw score dataframe
    stdNum = standard error numbers
    output data:
    output_data = result score with raw score field name + '_z'
    """
    # HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self):
        super(Zscore, self).__init__('zt')
        # self.model_name = 'zt'
        self.stdNum = 3
        self.maxRawscore = 150
        self.minRawscore = 0
        self._segtable = None
        self.__currentfield = None
        # create norm table
        self._samplesize = 100000    # cdf error is less than 0.0001
        self._normtable = exp_norm_table(self._samplesize, stdnum=4)
        self._normtable.loc[max(self._normtable.index), 'cdf'] = 1

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_parameters(self, std_num=3, rawscore_max=100, rawscore_min=0):
        self.stdNum = std_num
        self.maxRawscore = rawscore_max
        self.minRawscore = rawscore_min

    def check_parameter(self):
        if self.maxRawscore <= self.minRawscore:
            print('max raw score or min raw score error!')
            return False
        if self.stdNum <= 0:
            print('std number is error!')
            return False
        return True

    def run(self):
        # check data and parameter in super
        if not super().run():
            return
        self.output_data = self.input_data[self.field_list]
        self._segtable = self.__get_segtable(
            self.output_data,
            self.maxRawscore,
            self.minRawscore,
            self.field_list)
        for sf in self.field_list:
            print('start run...')
            st = time.clock()
            self._calczscoretable(sf)
            df = self.output_data.copy()
            print('zscore calculating1...')
            # new_score = [x if x in self._segtable.seg.values else -99 for x in df[sf]]
            df.loc[:, sf+'_zscore'] = df[sf].apply(lambda x: x if x in self._segtable.seg.values else -999)
            # df.loc[:, sf+'_zscore'] = new_score
            print('zscore calculating1...use time{}'.format(time.clock()-st))
            print('zscore calculating2...')
            df.loc[:, sf+'_zscore'] = df[sf + '_zscore'].replace(self._segtable.seg.values,
                                                                 self._segtable[sf+'_zscore'].values)
            self.output_data = df
            print('zscore transoform finished with {} consumed'.format(round(time.clock()-st, 2)))

    def _calczscoretable(self, sf):
        if sf+'_percent' in self._segtable.columns.values:
            self._segtable.loc[:, sf+'_zscore'] = \
                self._segtable[sf+'_percent'].apply(self.__get_zscore_from_normtable)
        else:
            print('error: not found field{}+"_percent"!'.format(sf))

    def __get_zscore_from_normtable(self, p):
        df = self._normtable.loc[self._normtable.cdf >= p - Zscore.MinError][['sv']].head(1).sv
        y = df.values[0] if len(df) > 0 else None
        if y is None:
            print('error: cdf value[{}] can not find zscore in normtable!'.format(p))
            return y
        return max(-self.stdNum, min(y, self.stdNum))

    @staticmethod
    def __get_segtable(df, maxscore, minscore, scorefieldnamelist):
        """no sort problem in this segtable usage"""
        seg = SegTable()
        seg.set_data(df, scorefieldnamelist)
        seg.set_parameters(segmax=maxscore, segmin=minscore, segsort='ascending')
        seg.run()
        return seg.output_data

    @staticmethod
    def get_normtable(stdnum=4, precise=4):
        cdf_list = []
        sv_list = []
        pdf_list = []
        cdf0 = 0
        scope = stdnum * 2 * 10**precise + 1
        for x in range(scope):
            sv = -stdnum + x/10**precise
            cdf = sts.norm.cdf(sv)
            pdf = cdf - cdf0
            cdf0 = cdf
            pdf_list.append(pdf)
            sv_list.append(sv)
            cdf_list.append(cdf)
        return pd.DataFrame({'pdf': pdf_list, 'sv': sv_list, 'cdf': cdf_list})

    def report(self):
        if type(self.output_data) == pd.DataFrame:
            print('output score desc:\n', self.output_data.describe())
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('parameters:')
        print('\tzscore stadard diff numbers:{}'.format(self.stdNum))
        print('\tmax score in raw score:{}'.format(self.maxRawscore))
        print('\tmin score in raw score:{}'.format(self.minRawscore))

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super().plot(mode)
        else:
            print('not support this mode!')


class Tscore(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模,平均数为50,标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出,是为了纪念推孟和桑代克
    对智力测验,尤其是提出智商这一概念所作出的巨大贡献。'''

    def __init__(self):
        super().__init__('t')
        # self.model_name = 't'

        self.rscore_max = 150
        self.rscore_min = 0
        self.tscore_std = 10
        self.tscore_mean = 50
        self.tscore_stdnum = 4

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_parameters(self, rawscore_max=150, rawscore_min=0, tscore_mean=50, tscore_std=10, tscore_stdnum=4):
        self.rscore_max = rawscore_max
        self.rscore_min = rawscore_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum

    def run(self):
        zm = Zscore()
        zm.set_data(self.input_data, self.field_list)
        zm.set_parameters(std_num=self.tscore_stdnum, rawscore_min=self.rscore_min,
                          rawscore_max=self.rscore_max)
        zm.run()
        self.output_data = zm.output_data
        namelist = self.output_data.columns
        for sf in namelist:
            if '_zscore' in sf:
                newsf = sf.replace('_zscore', '_tscore')
                self.output_data.loc[:, newsf] = \
                    self.output_data[sf].apply(lambda x: x * self.tscore_std + self.tscore_mean)

    def report(self):
        print('T-score by normal table transform report')
        print('-' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            print('    fields:', self.field_list)
            report_describe(
                self.input_data[self.field_list])
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.field_list]
            print('T-score desc:')
            print('    fields:', out_fields)
            report_describe(
                self.output_data[out_fields])
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('parameters:')
        print('\tzscore stadard deviation numbers:{}'.format(self.tscore_std))
        print('\tmax score in raw score:{}'.format(self.rscore_max))
        print('\tmin score in raw score:{}'.format(self.rscore_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)


class TscoreLinear(ScoreTransformModel):
    """Get Zscore by linear formula: (x-mean)/std"""
    def __init__(self):
        super().__init__('tzl')

        self.model_name = 'tzl'
        self.rawscore_max = 150
        self.rawscore_min = 0
        self.tscore_mean = 50
        self.tscore_std = 10
        self.tscore_stdnum = 4

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_parameters(self,
                       input_score_max=150,
                       input_score_min=0,
                       tscore_std=10,
                       tscore_mean=50,
                       tscore_stdnum=4):
        self.rawscore_max = input_score_max
        self.rawscore_min = input_score_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum

    def check_data(self):
        super().check_data()
        return True

    def check_parameter(self):
        if self.rawscore_max <= self.rawscore_min:
            print('raw score max and min error!')
            return False
        if self.tscore_std <= 0 | self.tscore_stdnum <= 0:
            print('t_score std number error:std={}, stdnum={}'.format(self.tscore_std, self.tscore_stdnum))
            return False
        return True

    def run(self):
        super().run()
        self.output_data = self.input_data[self.field_list]
        for sf in self.field_list:
            rmean, rstd = self.output_data[[sf]].describe().loc[['mean', 'std']].values[:, 0]
            self.output_data[sf + '_zscore'] = \
                self.output_data[sf].apply(
                    lambda x: min(max((x - rmean) / rstd, -self.tscore_stdnum), self.tscore_stdnum))
            self.output_data.loc[:, sf + '_tscore'] = \
                self.output_data[sf + '_zscore'].\
                apply(lambda x: x * self.tscore_std + self.tscore_mean)

    def report(self):
        print('TZ-score by linear transform report')
        print('-' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            report_describe(self.input_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            print('raw,T,Z score desc:')
            report_describe(self.output_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('parameters:')
        print('\tzscore stadard deviation numbers:{}'.format(self.tscore_std))
        print('\tmax score in raw score:{}'.format(self.rawscore_max))
        print('\tmin score in raw score:{}'.format(self.rawscore_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)


class LevelScore(ScoreTransformModel):
    """
    level score transform model
    default set to zhejiang project:
    level_ratio_table = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
    level_score_table = [100, 97, ..., 40]
    level_order = 'd'   # d: from high to low, a: from low to high
    """
    def __init__(self):
        super().__init__('level')
        __zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        self.approx_method_set = 'minmax, maxmin, nearmax, nearmin, near'

        self.input_score_max = 100
        self.input_score_min = 0
        self.level_ratio_table = [sum(__zhejiang_ratio[0:j+1])*0.01
                                  for j in range(len(__zhejiang_ratio))]
        self.level_score_table = [100-x*3 for x in range(len(self.level_ratio_table))]
        self.level_no = [x for x in range(1, len(self.level_ratio_table)+1)]
        self.level_order = 'd' if self.level_score_table[0] > self.level_score_table[-1] else 'a'
        self.approx_method = 'near'

        self.segtable = None
        self.output_data = None
        self.report_doc = ''

    def set_data(self, input_data=None, field_list=None):
        if isinstance(input_data, pd.DataFrame):
            self.input_data = input_data
        if isinstance(field_list, list):
            self.field_list = field_list
        elif isinstance(field_list, str):
            self.field_list = [field_list]
        else:
            print('error field_list: {}'.format(field_list))

    def set_parameters(self,
                       maxscore=None,
                       minscore=None,
                       level_ratio_table=None,
                       level_score_table=None,
                       approx_method=None):
        if isinstance(maxscore, int):
            if len(self.field_list) > 0:
                if maxscore >= max([max(self.input_data[f]) for f in self.field_list]):
                    self.input_score_max = maxscore
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set field_list first!')
        if isinstance(minscore, int):
            self.input_score_min = minscore
        if isinstance(level_ratio_table, list) or isinstance(level_ratio_table, tuple):
            self.level_ratio_table = [1-sum(level_ratio_table[0:j+1])*0.01
                                      for j in range(len(level_ratio_table))]
            if sum(level_ratio_table) != 100:
                print('ratio table is wrong, sum is not 100! sum={}'.format(sum(level_ratio_table)))
        if isinstance(level_score_table, list) or isinstance(level_score_table, tuple):
            self.level_score_table = level_score_table
        if len(self.level_ratio_table) != len(self.level_score_table):
            print('error level data set, ratio/score table is not same length!')
            print(self.level_ratio_table, '\n', self.level_score_table)
        self.level_no = [x for x in range(1, len(self.level_ratio_table)+1)]
        self.level_order = 'd' if self.level_score_table[0] > self.level_score_table[-1] else 'a'
        if approx_method in self.approx_method_set:
            self.approx_method = approx_method

    def run(self):
        if len(self.field_list) == 0:
            print('to set field_list first!')
            return
        seg = SegTable()
        seg.set_data(input_data=self.input_data,
                     field_list=self.field_list)
        seg.set_parameters(segmax=self.input_score_max,
                           segmin=self.input_score_min,
                           segsort=self.level_order)
        seg.run()
        self.segtable = seg.output_data
        self.__calc_level_table()
        self.output_data = self.input_data[self.field_list]
        self.report_doc = {}
        dtt = self.segtable
        for sf in self.field_list:
            dft = self.output_data.copy()
            dft[sf+'_percent'] = dft.loc[:, sf].replace(
                self.segtable['seg'].values, self.segtable[sf+'_percent'].values)
            dft[sf+'_percent'] = dft[sf+'_percent'].apply(
                lambda x: x if x in dtt['seg'] else -1)
            dft.loc[:, sf+'_level'] = dft.loc[:, sf].replace(
                self.segtable['seg'].values, self.segtable[sf + '_level'].values)
            dft[sf+'_level'] = dft[sf+'_level'].apply(
                lambda x: x if x in self.level_no else -1)
            dft.loc[:, sf+'_level_score'] = \
                dft.loc[:, sf+'_level'].apply(lambda x: self.level_score_table[int(x)-1]if x > 0 else x)
            # format to int
            dft = dft.astype({sf+'_level': int, sf+'_level_score': int})
            self.output_data = dft
            level_max = self.segtable.groupby(sf+'_level')['seg'].max()
            level_min = self.segtable.groupby(sf+'_level')['seg'].min()
            self.report_doc.update({sf: ['level({}):{}-{}'.format(j+1, x[0], x[1])
                                         for j, x in enumerate(zip(level_max, level_min))]})

    def __calc_level_table(self):
        for sf in self.field_list:
            self.segtable.loc[:, sf+'_level'] = self.segtable[sf+'_percent'].\
                apply(lambda x: self.__percent_map_level(1-x))
            self.segtable.astype({sf+'_level': int})

    def __percent_map_level(self, p):
        p_start = 0 if self.level_order == 'a' else 1
        for j, r in enumerate(self.level_ratio_table):
            logic = (p_start <= p <= r) if self.level_order == 'a' else (p_start >= p >= r)
            if logic:
                return self.level_no[j]
            p_start = r
        return self.level_no[-1]

    def report(self):
        print('Level-score transform report')
        print('=' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            report_describe(self.input_data[self.field_list])
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.segtable) == pd.DataFrame:
            print('raw,Level score desc:')
            report_describe(self.output_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('parameters:')
        print('\tmax score in raw score:{}'.format(self.input_score_max))
        print('\tmin score in raw score:{}'.format(self.input_score_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)

    def check_parameter(self):
        if self.input_score_max > self.input_score_min:
            return True
        else:
            print('raw score max value is less than min value!')
        return False

    def check_data(self):
        return super().check_data()

    def print_segtable(self):
        fs_list = ['seg']
        for ffs in self.field_list:
            fs_list += [ffs+'_count']
            fs_list += [ffs+'_percent']
            fs_list += [ffs+'_level']
        df = self.segtable.copy()
        for fs in fs_list:
            if 'percent' in fs:
                df[fs] = df[fs].apply(lambda x: round(x, 8))
        print(ptt.make_page(df=df[fs_list],
                            title='level score table for {}'.format(self.model_name),
                            pagelines=self.input_score_max+1))


class LevelScoreTao(ScoreTransformModel):
    """
    Level Score model from Tao BaiQiang
    high_level = rawscore().head(totalnum*0.01).mean
    intervals = [minscore, high_level*1/50], ..., [high_level, max_score]
    以原始分值切分，形成的分值相当于等距合并，粒度直接增加
    实质上失去了等级分数的意义
    本模型仍然存在高分区过度合并问题
    """

    def __init__(self):
        super().__init__('level')
        self.model_name = 'taobaiqiang'

        self.level_num = 50
        self.input_score_max = 100
        self.input_score_min = 0
        self.max_ratio = 0.01  # 1%
        self.input_data = pd.DataFrame()

        self.level_no = [x for x in range(self.level_num+1)]
        self.segtable = None
        self.level_dist_dict = {}  # fs: level_list, from max to min
        self.output_data = pd.DataFrame()

    def set_data(self, input_data=pd.DataFrame(), field_list=None):
        if len(input_data) > 0:
            self.input_data = input_data
        if isinstance(field_list, list) or isinstance(field_list, tuple):
            self.field_list = field_list

    def set_parameters(self,
                       maxscore=None,
                       minscore=None,
                       level_num=None,
                       ):
        if isinstance(maxscore, int):
            if len(self.field_list) > 0:
                if maxscore >= max([max(self.input_data[f]) for f in self.field_list]):
                    self.input_score_max = maxscore
                else:
                    print('error: maxscore is too little to transform score!')
            else:
                print('to set field_list first!')
        if isinstance(minscore, int):
            self.input_score_min = minscore
        if isinstance(level_num, int):
            self.level_num = level_num
        self.level_no = [x for x in range(self.level_num+1)]

    def run(self):
        self.run_create_level_dist_list()
        self.run_create_output_data()

    def run_create_level_dist_list(self):
        # approx_method = 'near'
        seg = SegTable()
        seg.set_parameters(segmax=self.input_score_max,
                           segmin=self.input_score_min,
                           segsort='d')
        seg.set_data(self.input_data,
                     self.field_list)
        seg.run()
        self.segtable = seg.output_data
        for fs in self.field_list:
            lastpercent = 0
            lastseg = self.input_score_max
            for ind, row in self.segtable.iterrows():
                curpercent = row[fs + '_percent']
                curseg = row['seg']
                if row[fs+'_percent'] > self.max_ratio:
                    if curpercent - self.max_ratio > self.max_ratio - lastpercent:
                        max_score = lastseg
                    else:
                        max_score = curseg
                    max_point = self.input_data[self.input_data[fs] >= max_score][fs].mean()
                    # print(fs, max_score, curseg, lastseg)
                    self.level_dist_dict.update({fs: round45i(max_point/self.level_num, 8)})
                    break
                lastpercent = curpercent
                lastseg = curseg
            # for lev in range(self.level_num):
            #     self.level_point_dict[fs].append(max_point - max_point/self.level_num * (lev+1))

    def run_create_output_data(self):
        dt = copy.deepcopy(self.input_data[self.field_list])
        for fs in self.field_list:
            dt.loc[:, fs+'_level'] = dt[fs].apply(lambda x: self.run__get_level_score(fs, x))
            dt2 = self.segtable
            dt2.loc[:, fs+'_level'] = dt2['seg'].apply(lambda x: self.run__get_level_score(fs, x))
            self.output_data = dt

    def run__get_level_score(self, fs, x):
        if x == 0:
            return x
        level_dist = self.level_dist_dict[fs]
        for i in range(self.level_num):
            minx = i * level_dist
            maxx = (i+1) * level_dist if i < self.level_num-1 else self.input_score_max
            if minx < x <= maxx:
                return i+1
        return -1

    def plot(self, mode='raw'):
        pass

    def report(self):
        report_describe(self.output_data[[f+'_level' for f in self.field_list]])

    def print_segtable(self):
        print(ptt.make_mpage(self.segtable))


# version 1.0.1 2018-09-24
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
        seg = SegTable()
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


def round45i(v: float, dec=0):
    u = int(v * 10 ** dec * 10)
    return (int(u / 10) + (1 if v > 0 else -1)) / 10 ** dec if (abs(u) % 10 >= 5) else int(u / 10) / 10 ** dec


# use scipy.stats descibe report dataframe info
def report_describe(df, decimal=4):
    """
    report statistic describe of a dataframe, with decimal digits = decnum
    峰度（Kurtosis）与偏态（Skewness）是量测数据正态分布特性的两个指标。
    峰度衡量数据分布的平坦度（flatness）。尾部大的数据分布峰度值较大。正态分布的峰度值为3。
        Kurtosis = 1/N * Sigma(Xi-Xbar)**4 / (1/N * Sigma(Xi-Xbar)**2)**2
    偏态量度对称性。0 是标准对称性正态分布。右（正）偏态表明平均值大于中位数，反之为左（负）偏态。
        Skewness = 1/N * Sigma(Xi-Xbar)**3 / (1/N * Sigma(Xi-Xbar)**2)**3/2
    :param
        dataframe: pandas DataFrame, raw data
        decnum: decimal number in report print
    :return(print)
        records
        min,max
        mean
        variance
        skewness
        kurtosis
    """

    def uf_list2str(listvalue, decimal):
        return ''.join([('{:' + '1.' + str(decimal) + 'f}  ').
                       format(round45i(x, decimal)) for x in listvalue])

    def uf_list2sqrt2str(listvalue, decimal):
        return uf_list2str([np.sqrt(x) for x in listvalue], decimal)

    pr = [[sts.pearsonr(df[x], df[y])[0]
           for x in df.columns] for y in df.columns]
    sd = sts.describe(df)
    cv = df.cov()
    print('\trecords: ', sd.nobs)
    print('\tpearson recorrelation:')
    for i in range(len(df.columns)):
        print('\t', uf_list2str(pr[i], 4))
    print('\tcovariance matrix:')
    for j in range(len(cv)):
        print('\t', uf_list2str(cv.iloc[j, :], 4))
    print('\tmin : ', uf_list2str(sd.minmax[0], 4))
    print('\tmax : ', uf_list2str(sd.minmax[1], 4))
    print('\tmean: ', uf_list2str(sd.mean, decimal))
    print('\tvar : ', uf_list2str(sd.variance, decimal))
    print('\tstd : ', uf_list2sqrt2str(sd.variance, decimal))
    print('\tskewness: ', uf_list2str(sd.skewness, decimal))
    print('\tkurtosis: ', uf_list2str(sd.kurtosis, decimal))
    dict = {'record': sd.nobs,
            'max': sd.minmax[1],
            'min': sd.minmax[0],
            'mean': sd.mean,
            'var': sd.variance,
            'cov': cv,
            'cor': pr,
            'skewness': sd.skewness,
            'kurtosis': sd.kurtosis,
            }
    return dict


def exp_norm_data(mean=70, std=10, maxvalue=100, minvalue=0, size=1000, decimal=6):
    """
    生成具有正态分布的数据，类型为 pandas.DataFrame, 列名为 sv
    create a score dataframe with fields 'score', used to test some application
    :parameter
        mean: 均值， std:标准差， maxvalue:最大值， minvalue:最小值， size:样本数
    :return
        DataFrame, columns = {'sv'}
    """
    # df = pd.DataFrame({'sv': [max(minvalue, min(int(np.random.randn(1)*std + mean), maxvalue))
    #                           for _ in range(size)]})
    df = pd.DataFrame({'sv': [max(minvalue,
                                  min(round45i(x, decimal) if decimal > 0 else int(round45i(x, decimal)),
                                      maxvalue))
                              for x in np.random.normal(mean, std, size)]})
    return df


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def exp_norm_table(size=400, std=1, mean=0, stdnum=4):
    """
    function
        生成正态分布量表
        create normal distributed data(pdf,cdf) with preset std,mean,samples size
        变量区间： [-stdNum * std, std * stdNum]
        interval: [-stdNum * std, std * stdNum]
    parameter
        变量取值数 size: variable value number for create normal distributed PDF and CDF
        分布标准差  std:  standard difference
        分布均值   mean: mean value
        标准差数 stdnum: used to define data range [-std*stdNum, std*stdNum]
    return
        DataFrame: 'sv':stochastic variable value,
                  'pdf': pdf value, 'cdf': cdf value
    """
    interval = [mean - std * stdnum, mean + std * stdnum]
    step = (2 * std * stdnum) / size
    varset = [mean + interval[0] + v*step for v in range(size+1)]
    cdflist = [sts.norm.cdf(v) for v in varset]
    pdflist = [sts.norm.pdf(v) for v in varset]
    ndf = pd.DataFrame({'sv': varset, 'cdf': cdflist, 'pdf': pdflist})
    return ndf
