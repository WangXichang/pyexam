# -*- utf-8 -*-
import pandas as pd
import pyex_stm as stm


# mode: plt-sd20, plt, tscore, zscore, L9
def model(name='plt-sd20',
          input_data=None,
          field_list=(),
          decimals=0,
          min_score=0,
          max_score=150):
    """model: 'plt-sd20', 'plt', 'zscore', 'tscore', 'tlinear', 'L9' """
    valid_model_names = ['plt-sd20', 'plt', 'zscore', 'tscore', 'tlinear', 'L9']
    if name not in valid_model_names:
        print('Invalid mode name: "{}" ! \nPlease choose from [{}]'.format(name, valid_model_names))
        return
    if not __check_para(input_data=input_data,
                        field_list=field_list):
        return

    if name.lower() == 'plt-sd20':
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

        score_percent_points = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]     # percent sequence
        output_score_points = [20, 30, 40, 50, 60, 70, 80, 90, 100]             # std=15.54(50), 15.61(40), 15.95(30)
        print('---shandong new gaokao score model---')
        print('score percent points: {}'.format(score_percent_points))
        print('output score  points: {}'.format(output_score_points))

        pltmodel = stm.PltScore()
        pltmodel.output_score_decimals = decimals
        pltmodel.set_data(input_data=input_data,
                          field_list=field_list)
        pltmodel.set_parameters(input_score_percent_list=score_percent_points,
                                output_score_points_list=output_score_points,
                                input_score_min=min_score,
                                input_score_max=max_score
                                )
        pltmodel.run()
        pltmodel.report()

        pltmodel.plot('out')
        # pltmodel.plot('raw')
        # pltmodel.plot('model')

        return pltmodel

    if name == 'plt':
        pltmodel = stm.PltScore()

        # score_percent_points = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        score_percent_points = [0, .15, .30, .50, .70, .85, 1.00]           # adjust ratio

        # output_score_points = [40, 50, 65, 80, 95, 110, 120]  # std=15.6576 at 65, std=15.04 at 50
        output_score_points = [20, 25, 40, 60, 80, 95, 100]     # std=19.6 at 40, std=17.55 at 25

        pltmodel.set_data(input_data=input_data,
                          field_list=field_list)
        pltmodel.set_parameters(input_score_percent_list=score_percent_points,
                                output_score_points_list=output_score_points,
                                input_score_max=max_score,
                                input_score_min=min_score)
        pltmodel.run()
        pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel

    if name == 'zscore':
        zm = stm.Zscore()
        zm.set_data(input_data=input_data,
                    input_field_list=field_list)
        zm.set_parameters(std_num=4, rawscore_max=max_score, rawscore_min=min_score)
        zm.run()
        zm.report()
        return zm

    if name == 'tscore':
        tm = stm.Tscore()
        tm.set_data(input_data=input_data,
                    input_field_list=field_list)
        tm.set_parameters(rawscore_max=150,
                          rawscore_min=0)
        tm.run()
        tm.report()
        return tm

    if name == 'tlinear':
        tm = stm.TscoreLinear()
        tm.set_data(input_data=input_data,
                    input_field_list=field_list)
        tm.set_parameters(input_score_max=max_score,
                          input_score_min=min_score)
        tm.run()
        tm.report()
        return tm

    if name.upper() == 'L9':
        tm = stm.L9score()
        tm.set_data(input_data=input_data,
                    field_list=field_list)
        tm.set_parameters(rawscore_max=max_score,
                          rawscore_min=min_score)
        tm.run()
        tm.report()
        return tm


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
