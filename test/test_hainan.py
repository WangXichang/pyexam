# coding: utf8


from stm import stm1
from test import test_stm


# hainan model problems:
# (1) max score = 900(300) at reatio==1.0 for ascending score order
#     but, min score may at 180-200(for 100-900) or 90-100(for 60-300)
#     with descending order, problem occur at max score.
#
# (2) weight may decrease to 1/3 if common subject score is 900,
#     it is reasonable if common subjects use raw score 150.
#
def test_hainan(num=1):
    if num == 1:
        # data1
        #    score point mean is bias to right(high), max==100(count==144), 0-4(count==0,0,0,1,1)
        test_data1 = test_stm.TestData(mean=60, std=14, size=60000)
        # use model100-900
        # score_order=='ascending', out_score_min==277, max==900, second_max=784
        #              'descending', out_score_max==784, second_min==101, 110, 123
        ht1a = stm1.run(name='hainan', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht1d = stm1.run(name='hainan', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        # use model60-300
        # score_order=='ascending', out_score_min==
        ht2a = stm1.run(name='hainan2', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht2d = stm1.run(name='hainan2', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')

    if num == 2:
        # data2
        test_data1 = test_stm.TestData(mean=50, std=14, size=60000)
        # use model100-900
        # score_order=='ascending', out_score_min==150(raw==0, count==12), max==900(count==11), second_max=856(count==6)
        #              'descending', out_score_max==861(count==9), min=100(raw=0, count==7), second_min==132,143 ,158
        ht1a = stm1.run(name='hainan', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht1d = stm1.run(name='hainan', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        # use model60-300
        # score_order=='ascending', out_score_min==69,73    max==300(100, 9), second_max==288(99, 5)
        #              'descending', out_score_max==288, second_min==60, 69, 73
        ht2a = stm1.run(name='hainan2', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='ascending')
        ht2d = stm1.run(name='hainan2', df=test_data1.df, cols=['km1', 'km2'], mode_score_order='descending')
        return ht1a, ht1d, ht2a, ht2d
