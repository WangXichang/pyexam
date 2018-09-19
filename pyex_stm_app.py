# -*- utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pyex_stm import PltScore, LevelScore, LevelScoreTao, Tscore, Zscore, TscoreLinear
# import pyex_stm as stm


# constant data
shandong_ratio = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]
shandong_level = [21, 31, 41, 51, 61, 71, 81, 91, 100]
zhejiang_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
shanghai_ratio = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
beijing_ratio = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
tianjin_ratio = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]


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
                                output_score_points_list=shandong_level,
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


def plot_model_ratio():
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


def __check_para(input_data, field_list):
    if type(input_data) != pd.DataFrame:
        print('no score dataframe given!')
        return False
    if len(field_list) == 0:
        print('no score field given!')
        return False
    if type(field_list) != list:
        print('inpurt_field_list is not list!')
        return False
    for f in field_list:
        if f not in input_data.columns:
            print('field {} is not in input_dataframe {}!'.format(f, input_data))
            return False
    return True
