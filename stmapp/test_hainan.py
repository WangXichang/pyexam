# coding: utf8


from stm import main
from stmapp import models_util as utl


# hainan model problems:
# (1) max score = 900(300) at reatio==1.0 for a score order
#     but, min score may at 180-200(for 100-900) or 90-100(for 60-300)
#     with d order, problem occur at max score.
#
# (2) weight may decrease to 1/3 if common subject score is 900,
#     it is reasonable if common subjects use raw score 150.
#
def test_hainan(num=1):
    r1, r2, r3, r4 = None, None, None, None
    if num == 1:
        # data1
        #    score point mean is bias to right(high), max==100(count==144), 0-4(count==0,0,0,1,1)
        test_data1 = utl.TestData(mean=60, std=14, size=60000)
        # use model100-900
        # score_order=='a', out_score_min==277, max==900, second_max=784
        #              'd', out_score_max==784, second_min==101, 110, 123
        r1 = main.run(model_name='hn300', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='a', verify=True)
        r2 = main.run(model_name='hn300', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='d', verify=True)
        # use model60-300
        # score_order=='a', out_score_min==
        r3 = main.run(model_name='h300plt1', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='a', verify=1)
        r4 = main.run(model_name='h300plt1', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='d', verify=1)

    if num == 2:
        # data2
        test_data1 = utl.TestData(mean=50, std=14, size=60000)
        # use model100-900
        # score_order=='a', out_score_min==150(raw==0, count==12), max==900(count==11), second_max=856(count==6)
        #              'd', out_score_max==861(count==9), min=100(raw=0, count==7), second_min==132,143 ,158
        r1 = main.run(model_name='h300', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='a', verify=1)
        r2 = main.run(model_name='h300', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='d', verify=1)
        # use model60-300
        # score_order=='a', out_score_min==69,73    max==300(100, 9), second_max==288(99, 5)
        #              'd', out_score_max==288, second_min==60, 69, 73
        r3 = main.run(model_name='h300plt2', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='a', verify=1)
        r4 = main.run(model_name='h300plt2', df=test_data1.df, cols=['km1', 'km2'], mode_sort_order='d', verify=1)

    return r1, r2, r3, r4
