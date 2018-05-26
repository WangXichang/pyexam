# -*- utf-8 -*-
import pandas as pd
import pyex_stm as stm


# mode: plt-sd20, plt, tscore, zscore, L9
def stm_app(name='plt-sd20',
            input_dataframe=None,
            score_field_list=(),
            decimal_place=0,
            min_score=0,
            max_score=150):
    valid_model_names = ['plt-sd20', 'plt', 'zscore', 'tscore', 'tlinear', 'L9']
    if name not in valid_model_names:
        print('Invalid mode name: "{}" ! \nPlease choose from [{}]'.format(name, valid_model_names))
        return
    if not __check_para(input_data=input_dataframe,
                        field_list=score_field_list):
        return

    if name.lower() == 'plt-sd20':
        print('---shandong new gaokao score model---')
        print('score percent points: {}'.format([0, .03, .10, .26, .50, .74, .90, .97, 1.00]))
        print('output score  points: {}'.format([20, 30, 40, 50, 60, 70, 80, 90, 100]))
        pltmodel = stm.PltScoreModel()
        score_percent_points = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]    # ajust ratio
        output_score_points = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # std=15
        pltmodel.output_score_decimals = decimal_place
        pltmodel.set_data(input_data=input_dataframe,
                          score_field_list=score_field_list)
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
        pltmodel = stm.PltScoreModel()

        # score_percent_points = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        score_percent_points = [0, .15, .30, .50, .70, .85, 1.00]           # adjust ratio

        # output_score_points = [40, 50, 65, 80, 95, 110, 120]  # std=15
        # output_score_points = [0, 15, 30, 50, 70, 85, 100]  # std=15
        output_score_points = [20, 25, 40, 60, 80, 95, 100]  # std=15

        pltmodel.set_data(input_data=input_dataframe,
                          score_field_list=score_field_list)
        pltmodel.set_parameters(input_score_percent_list=score_percent_points,
                                output_score_points_list=output_score_points,
                                input_score_max=max_score,
                                input_score_min=min_score)
        pltmodel.run()
        pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel

    if name == 'zscore':
        zm = stm.ZscoreByTable()
        zm.set_data(input_data=input_dataframe,
                    input_field_list=score_field_list)
        zm.set_parameters(std_num=4, rawscore_max=max_score, rawscore_min=min_score)
        zm.run()
        zm.report()
        return zm

    if name == 'tscore':
        tm = stm.TscoreByTable()
        tm.set_data(input_data=input_dataframe,
                    input_field_list=score_field_list)
        tm.set_parameters(rawscore_max=150,
                          rawscore_min=0)
        tm.run()
        tm.report()
        return tm

    if name == 'tlinear':
        tm = stm.TZscoreLinear()
        tm.set_data(input_data=input_dataframe,
                    input_field_list=score_field_list)
        tm.set_parameters(input_score_max=max_score,
                          input_score_min=min_score)
        tm.run()
        tm.report()
        return tm

    if name.upper() == 'L9':
        tm = stm.L9score()
        tm.set_data(input_data=input_dataframe,
                    input_field_list=score_field_list)
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
