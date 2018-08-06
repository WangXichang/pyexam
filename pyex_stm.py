# -*- utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pyex_seg as ps
import pyex_lib as pl
import scipy.stats as sts

# import warnings
# warnings.simplefilter('error')


def test(name='sdplt', df=None, field_list='',
         maxscore=100, minscore=0, decimals=0,
         percent_mode='maxmin'):
    """
    :param name: str, from ['plt', 'zscore', 'tscore', 'tlinear', 'l9']
    :param df: input dataframe
    :param field_list: score fields list in input dataframe
    :param decimals: output score decimals
    :return: model, object of ScoreTransformModel
    """
    if type(df) != pd.DataFrame:
        print('no score dataframe!')
        return
    else:
        scoredf = df
    if isinstance(field_list, str):
        field_list = [field_list]
    elif not isinstance(field_list, list):
        print('invalid field_list!')
        return

    # shandong new project score model
    if name == 'sdplt':
        # set model score percentages and endpoints
        # get approximate normal distribution
        # according to percent , test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #                        test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #                        test std=15.950713502      at 30    Zcdf(-30/std)=0.03
        #                        not real, but approximate normal distribution
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
        # -------------------------------------------------------------------------------------------------------------
        #     percent       0      0.03       0.10      0.26      0.50    0.74       0.90      0.97       1.00
        #   std/points      20      30         40        50        60      70         80        90         100
        #   15.54375    0.0050   0.0268       0.0991   [0.26000]   0    0.739(6)  0.9008989  0.97000    0.99496
        #   15.6060     0.0052   0.0273      [0.09999]  0.26083    0    0.73917   0.9000012  0.97272    0.99481
        #   15.9507     0.0061  [0.0299(5)]   0.10495   0.26535    0    0.73465   0.8950418  0.970(4)   0.99392
        # -------------------------------------------------------------------------------------------------------------
        # on the whole, fitting is approximate fine
        # p1: std scope in 15.54 - 15.95
        # p2: cut percent at 20, 100 is a little big
        # p3: percent at 30 is a bit larger than normal according to std=15.54375, same at 40
        # p4: max count at 60 estimated to [norm.pdf(0)=0.398942]/(sigma:pdf(50-60)=4.091)*0.24*total_number
        #     0.0234 * total_number
        #     200,000-->4680,   300,000 --> 7020

        rawpoints_sd = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]
        stdpoints_sd = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        low_rawscore_end = 0

        pltmodel = PltScore()
        pltmodel.output_score_decimals = 0
        pltmodel.set_data(input_data=scoredf, 
                          field_list=field_list)
        pltmodel.set_parameters(input_score_percent_list=rawpoints_sd,
                                output_score_points_list=stdpoints_sd,
                                input_score_max=maxscore,
                                input_score_min=minscore,
                                lookup_percent_mode=percent_mode,
                                define_low_endpoint=low_rawscore_end
                                )
        pltmodel.run()

        # pltmodel.report()
        # pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel

    if name == 'zscore':
        zm = Zscore()
        zm.set_data(input_data=scoredf, field_list=field_list)
        zm.set_parameters(std_num=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm
    if name == 'tscore':
        tm = Tscore()
        tm.set_data(input_data=scoredf, field_list=field_list)
        tm.set_parameters(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'tlinear':
        tm = TscoreLinear()
        tm.set_data(input_data=scoredf, field_list=field_list)
        tm.set_parameters(input_score_max=100, input_score_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'l9':
        tm = L9score()
        tm.set_data(input_data=scoredf, field_list=field_list)
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm


# Score Transform Model Interface
class ScoreTransformModel(object):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    本模块转换分数模型：
        Z分数非线性模型(Zscore)
        T分数非线性模型(Tscore）
        T分数线性模型（TscoreLinear)、
        标准九分数模型(L9score)
        分段线性转换分数山东省新高考改革转换分数模型（PltScore）
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

        self.sys_pricision_decimals = 6  #

    def set_data(self, input_data=None, field_list=None):
        raise NotImplementedError()

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if type(self.input_data) != pd.DataFrame:
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
            print('error mode={}, use valid mode: out or raw'.format(mode))
            return False
        return True

    def __plot_out_score(self):
        if not self.field_list:
            print('no field:{0} assign in {1}!'.format(self.field_list, self.input_data))
            return
        plt.figure(self.model_name + ' out score figure')
        labelstr = 'outscore: '
        for osf in self.output_data.columns.values:
            if '_' in osf:  # find sf_outscore field
                labelstr = labelstr + ' ' + osf
                plt.plot(self.output_data.groupby(osf)[osf].count())
                plt.xlabel(labelstr)
        return

    def __plot_raw_score(self):
        if not self.field_list:
            print('no field assign in rawdf!')
            return
        plt.figure('Raw Score figure')
        for sf in self.field_list:
            self.input_data.groupby(sf)[sf].count().plot(label=''.format(self.field_list))
        return


# model for linear score transform on some intervals
class PltScore(ScoreTransformModel):
    """
    PltModel:
    use linear standardscore transform from raw-score intervals
    to united score intervals
        # set model score percentages and endpoints
        # we get a near normal distribution
        # according to percent , test std=15.54374977       at 50    Zcdf(-10/std)=0.26
        #                        test std=15.60608295       at 40    Zcdf(-20/std)=0.10
        #                        test std=15.950713502        at 30    Zcdf(-30/std)=0.03
        #                        not real, but approximate normal distribution
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
        # p1: std scope in 15.5-16
        # p2: cut percent at 20, 100 is a little big
        # p3: percent at 30 is a bit larger than normal according to std=15.54375, same at 40
        # on the whole, fitting is approximate fine
    """

    def __init__(self):
        # intit input_df, input_output_data, output_df, model_name
        super(PltScore, self).__init__('plt')
        # self.model_name = 'plt'  # 'Pieceise Linear Transform Model'

        # new properties for linear segment stdscore
        self.input_score_percentage_points = []
        self.output_score_points = []
        
        # parameters
        self.lookup_percent_mode = 'minmax'
        self.define_low_endpoint = None

        # result
        self.reuslt_input_data_seg = pd.DataFrame()
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
                       lookup_percent_mode='minmax',
                       define_low_endpoint=None):
        """
        :param input_score_percent_list:
        :param output_score_points_list:
        :param input_score_min:
        :param input_score_max:
        :param lookup_percent_mode:  minmax, maxmin, nearmin, nearmax
        """
        if (type(input_score_percent_list) != list) | (type(output_score_points_list) != list):
            print('input score points or output score points is not list!')
            return
        if len(input_score_percent_list) != len(output_score_points_list):
            print('the number of input score points is not same as output score points!')
            return
        self.input_score_percentage_points = input_score_percent_list
        self.output_score_points = output_score_points_list
        self.input_score_min = input_score_min
        self.input_score_max = input_score_max
        self.lookup_percent_mode = lookup_percent_mode
        self.define_low_endpoint = define_low_endpoint

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
        self.reuslt_input_data_seg = seg.output_data

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
                'input_score_points': self.result_input_data_points,
                'coeff': self.result_coeff,
                'formulas': self.result_formula}

        self.output_report_doc = result_report_save
        self.output_data = result_dataframe.fillna(-1)

        print('used time:', time.time() - stime)
        print('-'*50)
        # run end

    def __get_plt_score(self, x):
        for i in range(1, len(self.output_score_points)):
            if x <= self.result_input_data_points[i]:
                y = self.result_coeff[i][0] * \
                    (x - self.result_coeff[i][1]) + self.result_coeff[i][2]
                # return self.score_round(y, self.output_score_decimals)
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

        # claculate _rawScorePoints
        if field in self.output_data.columns.values:
            print('-- get input score endpoints ...')
            self.result_input_data_points = self.__get_raw_score_points(field, self.lookup_percent_mode)
        else:
            print('score field({}) not in output_dataframe!'.format(field))
            print('must be in {}'.format(self.input_data.columns.values))
            return False

        # calculate Coefficients
        if not self.__getcoeff():
            return False

        return True

    def __get_raw_score_points(self, field, mode='minmax'):
        if mode not in 'minmax, maxmin, nearmax, nearmin':
            print('error mode {} !'.format(mode))
            raise TypeError
        if self.define_low_endpoint is not None:
            mode1 = 'defined_low_endpoint'
        else:
            mode1 = 'use_min_as_low_endpoint'
        score_points = [min(self.reuslt_input_data_seg['seg'][self.reuslt_input_data_seg[field+'_count']>0])]
        lastpercent = 0
        lastseg = self.input_score_min
        thisseg = 0
        percent_loc = 1
        for index, row in self.reuslt_input_data_seg.iterrows():
            p = row[field+'_percent']
            thisseg = row['seg']
            thispercent = self.input_score_percentage_points[percent_loc]
            if (p == 1) | (percent_loc == len(self.input_score_percentage_points)):
                score_points += [thisseg]
                break
            if mode in 'minmax, maxmin':
                if p > thispercent:
                    score_points += [lastseg] if mode == 'minmax' else [thisseg]
                    percent_loc += 1
            elif mode in 'nearmax, nearmin':
                if p > thispercent:
                    if (p-thispercent) < (thispercent - lastpercent):
                        score_points += [thisseg]
                    else:
                        score_points += [lastseg]
                    percent_loc += 1
                elif (p == thispercent) & (p == lastpercent):
                    if 'max' in mode:  # nearmax
                        score_points += [thisseg]
                    else:  # nearmin
                        score_points += [lastseg]
            lastseg = thisseg
            lastpercent = p
            if mode1 == 'defined_low_endpoint':
                if isinstance(self.define_low_endpoint, int):
                    score_points[0] = self.define_low_endpoint
                else:
                    print('define_low_endpoint must be int, not {}'.format(self.define_low_endpoint))
                    raise TypeError
        return score_points

    def __getcoeff(self):
        # formula: y = (y2-y1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1)/(x2 -x1)
        # print(self.result_input_data_points, self.output_score_points)
        if len(self.result_input_data_points) != len(self.output_score_points):
            print('error score points: {}'.format(self.result_input_data_points))
            return False

        for i in range(1, len(self.output_score_points)):
            if (self.result_input_data_points[i] - self.result_input_data_points[i - 1]) < 0.1**6:
                print('input score percent is not differrentiable or error order,{}-{}'.format(i, i-1))
                return False
            if self.result_input_data_points[i] - self.result_input_data_points[i - 1] > 0:
                coff = (self.output_score_points[i] - self.output_score_points[i - 1]) / \
                       (self.result_input_data_points[i] - self.result_input_data_points[i - 1])
            else:
                print('input score points[{0} - {1}] error!'.format(i-1, i))
                coff = 0
            y1 = self.output_score_points[i - 1]
            x1 = self.result_input_data_points[i - 1]
            coff = self.score_round(coff, self.sys_pricision_decimals)
            self.result_coeff[i] = [coff, x1, y1]

        return True

    def __pltrun(self, scorefieldname):
        if not self.__get_formula(scorefieldname):
            print('fail to initializing !')
            return

        # transform score
        self.output_data.loc[:, scorefieldname + '_plt'] = \
            self.input_data[scorefieldname].apply(self.__get_plt_score)  # , self.output_score_decimals)

        self._create_report()

    @staticmethod
    def round45i(v: float, dec=0):
        u = int(v * 10 ** dec * 10)
        return (int(u / 10) + (1 if v > 0 else -1)) / 10 ** dec if (abs(u) % 10 >= 5) else int(u / 10) / 10 ** dec

    @staticmethod
    def score_round(x, decimals=0):
        x_int = int(x * 10**(decimals+2))
        if decimals > 0:
            return np.floor(x_int/(10**2))/10**decimals \
                if np.mod(x_int, 100) < 50 else \
                (float(str((np.floor(x_int/(10**2))+1)/10**decimals))
                 if decimals > 0 else int(str((np.floor(x_int/(10**2))+1)/10**decimals)))
        elif decimals == 0:
            return int(np.floor(x_int/(10**2))) \
                if np.mod(x_int, 100) < 50 else \
                int(np.floor(x_int/(10**2)))+1
        else:
            return False

    def _create_report(self, field=''):
        self.result_formula = ['{0}*(x-{1})+{2}'.format(x[0], x[1], x[2])
                               for x in self.result_coeff.values()]
        self.output_report_doc = '---<< score field: {} >>---\n'.format(field)
        self.output_report_doc += 'input score percentage: {}\n'.format(self.input_score_percentage_points)
        self.output_report_doc += 'input score  endpoints: {}\n'.format(self.result_input_data_points)
        self.output_report_doc += 'output score endpoints: {}\n'.format(self.output_score_points)
        self.output_report_doc += '    transform formulas: {}\n'.format(self.result_formula)
        self.output_report_doc += '---'*30 + '\n\n'

    def report(self):
        print(self.output_report_doc)

    def plot(self, mode='raw'):
        if mode not in ['raw', 'out', 'model']:
            print('valid mode is: raw, out, model')
            print('mode:model describe the differrence of input and output score.')
            return
        if mode == 'model':
            self.__plotmodel()
        elif mode == 'pie':
            self.__plotpie()
        elif not super().plot(mode):
            print('mode {} is invalid'.format(mode))

    def __plotmodel(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams.update({'font.size':16})
        plen = len(self.result_input_data_points)
        flen = len(self.field_list)
        for i, fs in enumerate(self.field_list):
            plt.subplot(str(1)+str(flen)+str(i))
            plt.title(u'分段线性转换模型({})'.format(fs))
            result = self.result_dict[fs]
            input_points = result['input_score_points']
            raw_points = [(input_points[i-1], input_points[i]) for i in range(1, len(input_points))]
            out_points = [(self.output_score_points[i-1], self.output_score_points[i])
                          for i in range(1, len(input_points))]
            # print(raw_points, out_points)
            cross_points = []
            for i in range(len(raw_points)):
                if (out_points[i][0]-raw_points[i][0]) * (out_points[i][1]-raw_points[i][1]) < 0:
                    a, b, c = result['coeff'][i+1]
                    cross_points.append(round((a*b-c)/(a-1), 2))
                    # print(i, a,b,c)
            plt.xlim(input_points[0], input_points[-1])
            plt.ylim(self.output_score_points[0], self.output_score_points[plen - 1])
            plt.plot(input_points, self.output_score_points)
            plt.plot([input_points[0], input_points[-1]],
                     [input_points[0], input_points[-1]],
                     )
            if len(cross_points) > 0:
                for p in cross_points:
                    plt.plot([p, p], [0, p], '--')
                    plt.rcParams.update({'font.size': 12})
                    plt.text(p, p-2, '({})'.format(p))
            # plt.xlabel('{}'.format(fs))
        plt.show()
        return

    def __plotpie(self):
        plt.pie([],labels=[])


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
        self._normtable = pl.create_normaltable(self._samplesize, stdnum=4)
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
            print('zscore transoform finished with {} consumed'.format(round(time.clock()-st,2)))

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
            pl.report_stats_describe(
                self.input_data[self.field_list])
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            out_fields = [f+'_tscore' for f in self.field_list]
            print('T-score desc:')
            print('    fields:', out_fields)
            pl.report_stats_describe(
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
            pl.report_stats_describe(self.input_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            print('raw,T,Z score desc:')
            pl.report_stats_describe(self.output_data)
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


class L9score(ScoreTransformModel):
    """
    level 9 score transform model
    procedure: rawscore -> percent score(by segtable) -> 9 levels according cdf values:
    [0, 4%)->1, [4%, 11%)->2, [11%. 23%)->3, [23%, 40%)->4, [40%, 60%)->5
    [ 60%, 77%)->6, [77%, 89%)->7, [89%, 96%)->8, [97%, 100%]->9

    """
    def __init__(self):
        super().__init__('l9')
        # self.model_name = 'l9'

        self.rawscore_max = 100
        self.rawscore_min = 0
        self.levelscoretable = {1: [0, 0.04], 2: [0.04, 0.11], 3: [0.11, 0.23], 4: [0.23, 0.4], 5: [0.4, 0.6],
                                6: [0.6, 0.77], 7: [0.77, 0.89], 8: [0.89, 0.96], 9: [0.96, 1]}
        self.segtable = None
        self.output_data = None

    def set_data(self, input_data=None, field_list=None):
        self.input_data = input_data
        self.field_list = field_list

    def set_parameters(self, rawscore_max=100, rawscore_min=0):
        self.rawscore_max = rawscore_max
        self.rawscore_min = rawscore_min

    def run(self):
        # import py2ee_lib as pl
        seg = ps.SegTable()
        seg.set_data(input_data=self.input_data,
                     field_list=self.field_list)
        seg.set_parameters(segmax=self.rawscore_max, segmin=self.rawscore_min, segsort='ascending')
        seg.run()
        self.segtable = seg.output_data
        self.__calcscoretable()
        self.output_data = self.input_data[self.field_list]
        for sf in self.field_list:
            dft = self.output_data.copy()
            dft[sf+'_percent'] = dft.loc[:, sf].replace(self.segtable['seg'].values,
                                         self.segtable[sf+'_percent'].values)
            # self.output_data.loc[:, sf+'_percent'] = list(tes)
            dft.loc[:, sf+'_l9score'] = \
                dft.loc[:, sf].replace(self.segtable['seg'].values,
                                       self.segtable[sf+'_l9score'].values)
            self.output_data = dft

    def __calcscoretable(self):
        for sf in self.field_list:
            self.segtable.loc[:, sf+'_l9score'] = self.segtable[sf+'_percent'].\
                apply(lambda x: self.__percentmaplevel(x))

    def __percentmaplevel(self, p):
        for k in self.levelscoretable:
            if p < self.levelscoretable.get(k)[1]:
                return k
        return 9

    def report(self):
        print('L9-score transform report')
        print('=' * 50)
        if type(self.input_data) == pd.DataFrame:
            print('raw score desc:')
            pl.report_stats_describe(self.input_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.output_data) == pd.DataFrame:
            print('raw,L9 score desc:')
            pl.report_stats_describe(self.output_data)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print('data fields in rawscore:{}'.format(self.field_list))
        print('-' * 50)
        print('parameters:')
        print('\tmax score in raw score:{}'.format(self.rawscore_max))
        print('\tmin score in raw score:{}'.format(self.rawscore_min))
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)

    def check_parameter(self):
        if self.rawscore_max > self.rawscore_min:
            return True
        else:
            print('raw score max value is less than min value!')
        return False

    def check_data(self):
        return super().check_data()
