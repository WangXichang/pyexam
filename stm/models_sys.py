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
    STRATEGY = {strategy_name: value_str}

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
        退化区间映射：区间退化为单点情况，映射到转换分值的最大（to_max）、最小(map_to_min)、平均值(map_to_min)
          'mode_section_degraded':        ('to_max', 'map_to_min', 'map_to_mean'),
        消失区间处理：区间丢失情况，忽略(ignore)，向下增加一个点(next_one_point)，向下增加两个点(next_two_point)
          'mode_section_lost':            ('ignore', 'next_one_point', 'next_two_point'),

        注：1. 在上述策略的实现中，默认值为第一选择值。
            2. 目前已经实现的，被认为是最重要的, 有些使用了默认值。
            3. 分数转换中还存在一些其他策略，正在进一步的研究中。
"""


from collections import namedtuple
from stm import stmlib as slib


# model type
MODEL_TYPE_PLT = 'plt'      # piece-section linear transform
MODEL_TYPE_PPT = 'ppt'      # piece-point transform,     standard score transform
MODEL_TYPE_PGT = 'pgt'      # piece-grade transform,     standard score transform


hn900model = slib.get_norm_section_pdf(100, 900, 800, 4, True, 'ppt', 100, 'desceding')
hn300model = slib.get_norm_section_pdf(60, 300, 240, 4, True, 'ppt', 100, 'descending')
zscoremodel = slib.get_norm_section_pdf(-4, 4, 800, 4, True, 'ppt', 100, 'd')
tscoremodel = slib.get_norm_section_pdf(10, 90, 80, 4, True, 'ppt', 100, 'd')


# model parameters: type,   transform mode, in ['plt', 'ppt', 'tai']
#                           plt: zhejiang, shanghai, beijing, tianjin, shandong, guangdong, ss7, hn300plt1..plt3
#                           ppt: hn900, hn300, zscore, tscore
#                           tai: tai
#                  ratio,   tuple, list     # used to get raw score section in plt, to define out score percent in ppt
#                           sum==100
#                  section, tuple or list of tuple or list     # out score section
#                           len(section) == len(ratio);  p1>=p2 for each (p1, p2) in section
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
                                MODEL_TYPE_PGT,
                                [1, 99],                # only ratio[0]==1 is useful for set top score group
                                tuple((i+1, i+1) for i in range(15)),             # grade from 1 to 15
                                'taiwan grade score model, 1-15 levels, top_level = mean(top 1% scores)'
                                ),
    }


# choices = 4 * 2**5 * 3 * 2  = 768   ## prox, cumu, sort, section_
Strategy = {
    'mode_ratio_prox':              ('upper_min', 'lower_max', 'near_max', 'near_min'),
    'mode_ratio_cumu':              ('yes', 'no'),
    'mode_sort_order':              ('d', 'a'),                # d: descending, a: ascending
    'mode_section_point_first':     ('real', 'defined'),       # first point of first section, to defined maxmin score
    'mode_section_point_start':     ('step', 'share'),         # first point except first section
    'mode_section_point_last':      ('real', 'defined'),       # last point of last section, useful to type--ppt
    'mode_section_degraded':        ('to_max', 'to_min', 'to_mean'),
    'mode_section_lost':            ('real', 'zip'),                # not used in stm1
    }
