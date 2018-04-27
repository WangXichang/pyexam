# -*- utf-8 -*-


import pyex_stm as stm
import pandas as pd


# mode: plt-sd20, plt, tscore, zscore, L9
def test_model(name='plt-sd20',
               input_dataframe=None,
               input_fields_list=(),
               decimals=0,
               min_score=0,
               max_score=150):
    if type(input_dataframe) != pd.DataFrame:
        print('no score df given!')
        return

    if name == 'plt-sd20':
        pltmodel = stm.PltScoreModel()
        input_percentage_points = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]    # ajust ratio
        output_score_points = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # std=15

        pltmodel.output_score_decimals = decimals
        pltmodel.set_data(score_dataframe=input_dataframe,
                          score_fields_list=input_fields_list)
        pltmodel.set_parameters(input_percentage_points,
                                output_score_points,
                                input_score_min=min_score,
                                input_score_max=max_score)
        pltmodel.run()

        pltmodel.report()
        # pltmodel.plot('raw')   # plot raw score figure, else 'out', 'model'
        # pltmodel.plot('out')
        # pltmodel.plot('model')

        return pltmodel

    if name == 'plt':
        pltmodel = stm.PltScoreModel()
        # rawpoints = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        input_percentage_points = [0, .15, .30, .50, .70, .85, 1.00]    # ajust ratio
        # stdpoints = [40, 50, 65, 80, 95, 110, 120]  # std=15
        # stdpoints = [0, 15, 30, 50, 70, 85, 100]  # std=15
        output_score_points = [20, 25, 40, 60, 80, 95, 100]  # std=15

        pltmodel.set_data(score_dataframe=input_dataframe, score_fields_list=input_fields_list)
        pltmodel.set_parameters(input_percentage_points, output_score_points)
        pltmodel.run()
        pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel

    if name == 'zscore':
        zm = stm.ZscoreByTable()
        zm.set_data(input_dataframe, input_fields_list)
        zm.set_parameters(std_num=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm
    if name == 'tscore':
        tm = stm.TscoreByTable()
        tm.set_data(input_dataframe, input_fields_list)
        tm.set_parameters(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'tlinear':
        tm = stm.TZscoreLinear()
        tm.set_data(input_dataframe, input_fields_list)
        tm.set_parameters(input_score_max=100, input_score_min=0)
        tm.run()
        tm.report()
        return tm
    if name.upper() == 'L9':
        tm = stm.L9score()
        tm.set_data(input_dataframe, input_fields_list)
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
