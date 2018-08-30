# -*- utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
import copy
import time
import pyex_seg as ps
import pyex_lib as pl
import scipy.stats as sts
import seaborn as sbn
import pyex_ptt as ptt

# import warnings
# warnings.simplefilter('error')

# constant data
shandong_ratio = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]
shandong_level = [21, 31, 41, 51, 61, 71, 81, 91, 100]
# shandong_level = [20, 30, 40, 50, 60, 70, 80, 90, 100]
zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
shanghai_ratio = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
beijing_ratio = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
tianjin_ratio = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]


# interface to use model for some typical application
def usemodel(name='shandong',
             df=None,
             field_list='',
             maxscore=100,
             minscore=0,
             decimals=0,
             approx_method='nearmin'
             ):
    """
    :param name: str, 'shandong', 'shanghai', 'shandong', 'beijing', 'tianjin', 'zscore', 'tscore', 'tlinear'
    :param df: input dataframe
    :param field_list: score fields list in input dataframe
    :param maxscore: max value in raw score
    :param minscore: min value in raw score
    :param decimals: output score decimals
    :param approx_method: maxmin, minmax, nearmin, nearmax
    :return: model
    """

    # valid name
    name_set = 'zhejiang, shanghai, shandong, beijing, tianjin, ' \
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
        pltmodel.output_score_decimals = 0
        pltmodel.set_data(input_data=input_data,
                          field_list=field_list)
        pltmodel.set_parameters(input_score_percent_list=shandong_ratio,
                                output_score_points_list=shandong_level,
                                input_score_max=maxscore,
                                input_score_min=minscore,
                                approx_mode=approx_method,
                                use_minscore_as_start_endpoint=True,
                                decimals=decimals
                                )
        pltmodel.run()
        return pltmodel

    if name == 'zhejiang':
        # estimate: std = 14, mean=70
        level_score_table = [100-x*3 for x in range(21)]
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
        level_score = [70-j*3 for j in range(11)]
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
        level_score = [100-j*3 for j in range(21)]
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
        level_score = [100-j*3 for j in range(21)]
        m = LevelScore()
        m.model_name = 'tianjin'
        m.set_data(input_data=input_data, field_list=field_list)
        m.set_parameters(level_ratio_table=tianjin_ratio,
                         level_score_table=level_score,
                         maxscore=maxscore,
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


def plot_model_ratio():
    plt.figure('model ratio distribution')
    plt.rcParams.update({'font.size': 16})
    plt.subplot(231)
    plt.bar(range(8), [shandong_ratio[j] - shandong_ratio[j - 1] for j in range(1, 9)])
    plt.title('shandong model')

    plt.subplot(232)
    plt.bar(range(11), [shanghai_ratio[-j-1] for j in range(11)])
    plt.title('shanghai model')

    plt.subplot(233)
    plt.bar(range(21), [zhejiang_ratio[-j-1] for j in range(len(zhejiang_ratio))])
    plt.title('zhejiang model')

    plt.subplot(234)
    plt.bar(range(21), [beijing_ratio[-j-1] for j in range(len(beijing_ratio))])
    plt.title('beijing model')

    plt.subplot(235)
    plt.bar(range(21), [tianjin_ratio[-j-1] for j in range(len(tianjin_ratio))])
    plt.title('tianjin model')


def report_mean_median_mode(df, field_list, with_zero=False):
    from scipy.stats import mode
    from numpy import mean, median
    for field in field_list:
        if not with_zero:
            st = (mean(df[df[field] > 0][field]),
                  median(df[df[field] > 0][field]),
                  mode(df[df[field] > 0][field])[0][0],
                  mode(df[df[field] > 0][field])[1][0])
            print('field:{}(not with zero) '
                  'mean={:.2f}, median={:.2f}, mode={} modecount={}'.
                  format(field, st[0], st[1], st[2], st[3]))
        else:
            st = (
                mean(df[field]),
                median(df[field]),
                mode(df[field])[0][0],
                mode(df[field])[1][0])
            print('field:{}(with zero) '
                  'mean={:.2f}, median={:.2f}, mode={} modecount={}'.
                  format(field, st[0], st[1], st[2], st[3]))


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
        self.output_score_decimals = 0
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
        self.output_score_decimals = 0

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
        if len(input_score_percent_list) != len(output_score_points_list):
            print('the number of input score points is not same as output score points!')
            return
        if isinstance(decimals, int):
            self.output_score_decimals = decimals
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
        import pyex_seg as psg
        seg = psg.SegTable()
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
            score_list = df2[fs].apply(self.__get_plt_score, self.output_score_decimals)
            self.output_data.loc[:, (fs + '_plt')] = score_list
            self._create_report(fs)
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
        for i in range(1, len(self.output_score_points)):
            if x <= self.result_input_data_points[i]:
                y = self.result_coeff[i][0] * \
                    (x - self.result_coeff[i][1]) + self.result_coeff[i][2]
                return self.round45i(y, self.output_score_decimals)
        return -1

    def __get_formula(self, field):
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
        if len(self.input_score_percentage_points) != len(self.output_score_points):
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
            print('score field({}) not in output_dataframe!'.format(field))
            print('must be in {}'.format(self.input_data.columns.values))
            return False

        # step 2
        # calculate Coefficients
        if not self.__getcoeff():
            return False

        return True

    def __get_raw_score_points(self, field, mode='minmax'):
        if mode not in 'minmax, maxmin, nearmax, nearmin':
            print('error mode {} !'.format(mode))
            raise TypeError
        # the lowest end of inscore_points
        # if self.use_minscore_as_rawscore_start_endpoint:
        #     score_points = [self.input_score_min]
        # else:
        #     score_points = [min(self.input_data[field])]

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

    def __getcoeff(self):
        # formula: y = (y2-y1-1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1-1)/(x2 -x1)

        if len(self.result_input_data_points) != len(self.output_score_points):
            print('error score points: {}'.format(self.result_input_data_points))
            return False

        for i in range(1, len(self.output_score_points)):
            if (self.result_input_data_points[i] - self.result_input_data_points[i - 1]) < 0.1**6:
                print('input score percent is not differrentiable or error order,{}-{}'.format(i, i-1))
                return False
            if self.result_input_data_points[i] - self.result_input_data_points[i - 1] > 0:
                if i < len(self.output_score_points) -1:
                    # [xi, x(i+1) - 1] -> [yi, y(i+1) - 1]
                    coff = (self.output_score_points[i] - self.output_score_points[i - 1]-1) / \
                           (self.result_input_data_points[i] - self.result_input_data_points[i - 1]-1)
                else:
                    # [xi, xmax] -> [yi, ymax]
                    coff = (self.output_score_points[i] - self.output_score_points[i - 1]) / \
                           (self.result_input_data_points[i] - self.result_input_data_points[i - 1])
            else:
                print('input score points[{0} - {1}] error!'.format(i-1, i))
                coff = -1
            y1 = self.output_score_points[i - 1]
            x1 = self.result_input_data_points[i - 1]
            coff = self.round45i(coff, self.sys_pricision_decimals)
            self.result_coeff[i] = [coff, x1, y1]

        return True

    def __pltrun(self, scorefieldname):
        # create formula
        if not self.__get_formula(scorefieldname):
            print('fail to initializing !')
            return
        # transform score
        self.output_data.loc[:, scorefieldname + '_plt'] = \
            self.input_data[scorefieldname].apply(self.__get_plt_score)  # , self.output_score_decimals)
        # create report
        self._create_report()

    def get_plt_score_from_formula(self, field, x, decimal=0):
        if field not in self.field_list:
            print('invalid field name {} not in {}'.format(field, self.field_list))
        coeff = self.result_dict[field]['coeff']
        maxkey = len(coeff)
        result = -1
        for i in range(1, maxkey+1):
            if x < coeff[i][1]:
                result = self.round45i(coeff[i-1][0]*(x-coeff[i-1][1]) + coeff[i-1][2], decimal)
                break
            if i == maxkey:
                result = self.round45i(coeff[i][0]*(x-coeff[i][1]) + coeff[i][2], decimal)
        result = min(self.input_score_max, result)
        return result

    @staticmethod
    def round45i(v: float, dec=0):
        u = int(v * 10 ** dec * 10)
        return (int(u / 10) + (1 if v > 0 else -1)) / 10 ** dec if (abs(u) % 10 >= 5) else int(u / 10) / 10 ** dec

    def _create_report(self, field=''):
        self.result_formula = ['{0}*(x-{1})+{2}'.format(x[0], x[1], x[2])
                               for x in self.result_coeff.values()]
        self.output_report_doc = '---<< score field: {} >>---\n'.format(field)
        self.output_report_doc += 'input score percentage: {}\n'.format(self.input_score_percentage_points)
        self.output_report_doc += 'input score  endpoints: {}\n'.format(self.result_input_data_points)
        self.output_report_doc += 'output score endpoints: {}\n'.format(self.output_score_points)
        for i, fs in enumerate(self.result_formula):
            if i == 0:
                self.output_report_doc += '    transform formulas: {}\n'.format(fs)
            else:
                self.output_report_doc += '                        {}\n'.format(fs)
        self.output_report_doc += '---'*30 + '\n\n'

    def report(self):
        print(self.output_report_doc)

    def plot(self, mode='raw'):
        if mode not in ['raw', 'out', 'model', 'shift']:
            print('valid mode is: raw, out, model,shift')
            # print('mode:model describe the differrence of input and output score.')
            return
        if mode == 'model':
            self.__plotmodel()
        elif mode == 'shift':
            self.__plotshift()
        elif not super().plot(mode):
            print('mode {} is invalid'.format(mode))

    def __plotmodel(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams.update({'font.size': 8})
        for i, fs in enumerate(self.field_list):
            result = self.result_dict[fs]
            input_points = result['input_score_points']

            # disp distribution with input_points label
            # plt.subplot(131)
            plt.figure(fs+'_raw')
            plt.yticks([])
            input_points_count = self.segtable[self.segtable.seg.isin(
                input_points)][fs+'_count'].values
            # plt.plot(self.segtable['seg'], self.segtable[fs+'_count'])
            sbn.distplot(self.input_data[fs])
            # plt.rcParams.update({'font.size': 8})
            plt.xlabel(u'原始分数')
            for p, q in zip(input_points, input_points_count):
                plt.plot([p, p], [0, 0.1], '--')
                plt.text(p, -0.001, '{}'.format(int(p)))

            # 分段线性转换模型
            # plt.subplot(132)
            plt.figure(fs+'_plt')
            plt.title(u'分段线性正态转换模型')
            plt.xlim(input_points[0], input_points[-1])
            plt.ylim(self.output_score_points[0], self.output_score_points[-1])
            plt.plot(input_points, self.output_score_points)
            # plt.plot([input_points[0], input_points[-1]], [input_points[0], input_points[-1]])
            plt.rcParams.update({'font.size': 8})
            # plt.text(95, 16, '100')
            plt.xlabel(u'原始分数')
            plt.ylabel(u'转换分数')
            # plt.rcParams.update({'font.size': 8})
            for p, q in zip(input_points, self.output_score_points):
                plt.plot([p, p], [0, q], '--')
                plt.plot([0, p], [q, q], '--')
                plt.text(p, self.output_score_points[0]-2, '{}'.format(int(p)))

            # plt score
            # plt.subplot(133)
            plt.figure(fs+'_out')
            plt.yticks([])
            sbn.distplot(self.output_data[fs+'_plt'])
            plt.xlabel(u'转换分数')

        plt.show()
        return

    def __plotshift(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plen = len(self.result_input_data_points)
        flen = len(self.field_list)
        for i, fs in enumerate(self.field_list):
            plt.subplot(str(1)+str(flen)+str(i))
            plt.rcParams.update({'font.size': 20})
            plt.title(u'分段线性转换模型({})'.format(fs))
            plt.rcParams.update({'font.size': 16})
            plt.xlabel(u'原始分数')
            plt.rcParams.update({'font.size': 16})
            plt.ylabel(u'转换分数')
            result = self.result_dict[fs]
            input_points = result['input_score_points']
            raw_points = [(input_points[i-1], input_points[i]) for i in range(1, len(input_points))]
            out_points = [(self.output_score_points[i-1], self.output_score_points[i])
                          for i in range(1, len(input_points))]
            cross_points = []
            for j in range(len(raw_points)):
                if (out_points[j][0]-raw_points[j][0]) * (out_points[j][1]-raw_points[j][1]) < 0:
                    a, b, c = result['coeff'][j+1]
                    cross_points.append(round((a*b-c)/(a-1), 1))
            plt.xlim(input_points[0], input_points[-1])
            plt.ylim(self.output_score_points[0], self.output_score_points[plen - 1])
            plt.plot(input_points, self.output_score_points)
            plt.plot([input_points[0], input_points[-1]],
                     [input_points[0], input_points[-1]]
                     )
            if len(cross_points) > 0:
                for p in cross_points:
                    plt.plot([p, p], [0, p], '--')
                    plt.rcParams.update({'font.size': 12})
                    plt.text(p, p-2, '({})'.format(p))
        plt.show()
        return

    # from formula
    # def get_plt_score(self, fs, x):
    #     coeff = self.result_dict[fs]['coeff']
    #     input_points = self.result_dict[fs]['input_score_points']
    #     for i in range(1, len(self.output_score_points)):
    #         if x <= input_points[i]:
    #             y = coeff[i][0] * (x - coeff[i][1]) + coeff[i][2]
    #             return self.round45i(y, self.output_score_decimals)
    #     return -1

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
    HighPrecise = 0.9999999
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
        self._normtable = pl.exp_norm_table(self._samplesize, stdnum=4)
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
        seg = ps.SegTable()
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
            pl.report_describe(
                self.input_data[self.field_list])
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.field_list]
            print('T-score desc:')
            print('    fields:', out_fields)
            pl.report_describe(
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
            pl.report_describe(self.input_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            print('raw,T,Z score desc:')
            pl.report_describe(self.output_data)
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
        seg = ps.SegTable()
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
                dft.loc[:, sf+'_level'].\
                    apply(lambda x: self.level_score_table[int(x)-1]if x > 0 else x)
            # format to int
            dft = dft.astype({sf+'_level':int, sf+'_level_score': int})
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
            pl.report_describe(self.input_data[self.field_list])
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.segtable) == pd.DataFrame:
            print('raw,Level score desc:')
            pl.report_describe(self.output_data)
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
    """
    def __init__(self):
        super().__init__('level')
