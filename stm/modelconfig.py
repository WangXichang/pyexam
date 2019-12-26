# coding: utf-8


"""
    [CONSTANTS] 模块中的常量

    分数转换方式：
    MODEL_TYPE = {'plt',    # 分段线性转换 piecewise linear transform
                  'ppt'     # 逐点转换 piecewise point transform
                  }

    模型参数
    MODEL_SETTING_DICT = {'type': str := 'plt' or 'ppt',
                          'name': str := 'shandong', 'shanghai', ...
                          'ratio': list := percent value for segments
                          'seg': list := output score segments or points
                          'desc': str := test to describe model
                          }
        各等级分数转换比例设置，用于定义模型
        比例的设置从高分段到低分段，与分段设置相对应
        例如：
        ZHEJIANG_RATIO   =  [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        ZHEJIANG_SECTION =  tuple((x, x) for x in range(100, 39, -3)),
        SHANDONG_RATIO =    [3, 7, 16, 24, 24, 16, 7, 3]
        SHANDONG_SECTION =  [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
        SS7_RATIO =         [2, 13, 35, 35, 15]
        SS7_SECTION =       [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]

    转换算法策略
    MODEL_STRATEGY_DICT = {key_name_str: value_str}   # some choice assigned in value_str, seprated by comma

        目前分析实现的算法策略及选择值：

        比例逼近策略
         'mode_ratio_prox':               ('upper_min', 'lower_max', 'near_max', 'near_min'),
        比例累计策略
         'mode_ratio_cumu':               ('yes', 'no'),
        搜索比例值的分数顺序
         'model_score_sort':              ('descending', 'ascending'),
        第一端点确定：区间第一个端点映射到转换分数最高(低)值(real)、卷面最高（低）值(defined)
          'mode_section_point_first':     ('real', 'defined')
        开始端点确定：区间开始端点映射到上一个区间的下一个值(step)、使用上一个区间端点（share）
          'mode_section_point_start':     ('step', 'share')
        最后端点确定：区间最后端点映射到实际最高（低）分(real)、卷面最高（低）分(defined)
          'mode_section_point_last':      ('real', 'defined')
        退化区间映射：区间退化为单点情况，映射到转换分值的最大（map_to_max）、最小(map_to_min)、平均值(map_to_min)
          'mode_section_degraded':        ('map_to_max', 'map_to_min', 'map_to_mean'),
        消失区间处理：区间丢失情况，忽略(ignore)，向下增加一个点(next_one_point)，向下增加两个点(next_two_point)
          'mode_section_lost':            ('ignore', 'next_one_point', 'next_two_point'),

        在上述策略的实现中，默认值为第一选择值。

        上述策略是目前已经实现的，被认为是最重要的
        在转换中还存在一些策略，可以进一步研究。
"""


from collections import namedtuple
from stm import modellib as mlib


# model type
MODEL_TYPE_PLT = 'plt'      # piecewise linear transform
MODEL_TYPE_PPT = 'ppt'      # standard score transform


hn900model = mlib.ModelAlgorithm.get_section_pdf(100, 900, 800, 4, True, 'ppt', 100, 'desceding')
hn300model = mlib.ModelAlgorithm.get_section_pdf(60, 300, 240, 4, True, 'ppt', 100, 'descending')
zscoremodel = mlib.ModelAlgorithm.get_section_pdf(-4, 4, 8000, 4, True, 'ppt', 100, 'd')
tscoremodel = mlib.ModelAlgorithm.get_section_pdf(10, 90, 80, 4, True, 'ppt', 100, 'd')

# model including: type,    transfrom mode, in ['plt', 'ppt', 'tai']
#                  ratio,   used to get raw score section in plt, to define out score percent in ppt
#                  section, out score section
#                  desc,    describing model
ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'section', 'desc'])
Models = {
    'zhejiang':     ModelFields(MODEL_TYPE_PLT,
                                (1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1),
                                tuple((x, x) for x in range(100, 39, -3)),
                                'piecewise linear transform model'
                                ),
    'shanghai':     ModelFields(MODEL_TYPE_PLT,
                                (5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5),
                                tuple((x, x) for x in range(70, 39, -3)),
                                'piecewise linear transform model'
                                ),
    'beijing':      ModelFields(MODEL_TYPE_PLT,
                                (1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1),
                                tuple((100-i*3, 100-i*3) for i in range(21)),
                                'piecewise linear transform model'),
    'tianjin':      ModelFields(MODEL_TYPE_PLT,
                                (2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1),
                                tuple((100-i*3, 100-i*3) for i in range(21)),
                                'piecewise linear transform model'
                                ),
    'shandong':     ModelFields(MODEL_TYPE_PLT,
                                (3, 7, 16, 24, 24, 16, 7, 3),
                                tuple((100-i*10, 100-i*10-9) for i in range(8)),
                                'piecewise linear transform model'
                                ),
    'guangdong':    ModelFields(MODEL_TYPE_PLT,
                                (17, 33, 33, 15, 2),
                                ((100, 83), (82, 71), (70, 59), (58, 41), (40, 30)),
                                'piecewise linear transform model'
                                ),
    'ss7':          ModelFields(MODEL_TYPE_PLT,
                                (15, 35, 35, 13, 2),
                                ((100, 86), (85, 71), (70, 56), (55, 41), (40, 30)),
                                'piecewise linear transform model'
                                ),
    'hn900':        ModelFields(MODEL_TYPE_PPT,
                                hn900model.pdf,
                                hn900model.section,
                                'standard score model: piecewise point transform'),
    'hn300':        ModelFields(MODEL_TYPE_PPT,
                                hn300model.pdf,
                                hn300model.section,
                                'standard score model: piecewise point transform'
                                ),
    'hn300plt1':    ModelFields(
                                MODEL_TYPE_PLT,
                                (0.14, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.14),
                                tuple((x, x-30+1) if x > 90 else (x, x-30) for x in range(300, 60, -30)),
                                # ((300, 271), (270, 241), ... , (120, 91), (90, 60)),
                                'piecewise linear transform model'
                                ),
    'hn300plt2':    ModelFields(
                                MODEL_TYPE_PLT,
                                (0.2, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.2),
                                tuple((x, x - 30 + 1) if x > 90 else (x, x - 30) for x in range(300, 60, -30)),
                                'piecewise linear transform model'
                                ),
    'hn300plt3':    ModelFields(
                                MODEL_TYPE_PLT,
                                (1, 2, 14, 33, 33, 14, 2, 1),
                                tuple((x, x - 30 + 1) if x > 90 else (x, x - 30) for x in range(300, 60, -30)),
                                'piecewise linear transform model with ratio-segment'
                                ),
    'zscore':       ModelFields(MODEL_TYPE_PPT,
                                zscoremodel.pdf,
                                zscoremodel.section,
                                'piecewise linear transform model with ratio-segment'
                                ),
    'tscore':       ModelFields(MODEL_TYPE_PPT,
                                tscoremodel.pdf,
                                tscoremodel.section,
                                'piecewise linear transform model with ratio-segment'
                                ),
    'tai':          ModelFields(
                                'tai',
                                [1 for _ in range(15)],     # only first==1 is useful
                                tuple((i+1, i+1) for i in range(15)),
                                'piecewise linear transform model with ratio-segment'
                                ),
    }


# choice = 4 * 2**5  * 3**2 = 1152    ## prox, cumu, sort, section_
Strategies = {
    'mode_ratio_prox':              ('upper_min', 'lower_max', 'near_max', 'near_min'),
    'mode_ratio_cumu':              ('yes', 'no'),
    'mode_sort_order':              ('descending', 'ascending'),
    'mode_section_point_first':     ('real', 'defined'),        # first point of first section
    'mode_section_point_start':     ('step', 'share'),          # first point except first section
    'mode_section_point_last':      ('real', 'defined'),        # last point of last section, useful to type--ppt
    'mode_section_degraded':        ('map_to_max', 'map_to_min', 'map_to_mean'),
    'mode_section_lost':            ('ignore', 'next_one_point', 'next_two_point'),
    }
# to add in future
# MODEL_STRATEGIES_RESERVE_DICT = {
#       'mode_score_empty': ('use', 'jump'),          # ** consider to deprecated, processed in other strategies
#       'mode_score_rmin_to_min': ('ignore', 'yes'),  # real raw score min value to out score min value,
#                                                     # case: sort by 'a', standard score mode
#       'mode_score_rmax_to_max': ('ignore', 'yes'),  # real raw score max value to out score max value,
#                                                     # case: top ratio large, sort by 'a', standard score mode
# }
