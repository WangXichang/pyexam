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
        各省市等级分数转换比例设置，用于定义模型
        CONST_ZHEJIANG_RATIO = [1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1]
        CONST_SHANGHAI_RATIO = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5]
        CONST_BEIJING_RATIO = [1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1]
        CONST_TIANJIN_RATIO = [2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1]
        CONST_SHANDONG_RATIO = [3, 7, 16, 24, 24, 16, 7, 3]
        CONST_SHANDONG_SEGMENT = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
        CONST_SS7_RATIO = [2, 13, 35, 35, 15]
        CONST_SS7_SEGMENT = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]

    转换算法策略
    MODEL_STRATEGY_DICT = {key_name_str: value_str}   # some choice assigned in value_str, seprated by comma

        目前分析实现的算法策略名称及选择值：

        比例逼近策略
        * 'mode_ratio_prox':        ('upper_min', 'lower_max', 'near_max', 'near_min'),
        比例累计策略
        * 'mode_ratio_cumu':        ('yes', 'no'),
        搜索比例值的分数顺序
        * 'model_score_sort':       ('ascending', 'descending'),
        分数满分值是否映射到转换分数最大值，零分是否映射到最小值，实际最高分是否映射到最大值
          'mode_score_full_to_max': ('ignore', 'yes'),    # not for empty, but for ratio
          'mode_score_zero_to_min': ('no', 'yes'),        # ...
          'mode_score_max_to_max':  ('ignore', 'yes'),    # max raw score to max out score
          'mode_score_empty':       ('ignore', 'map_to_up', 'map_to_low'),
        区间单点情况，映射到最大、最小、平均值
        * 'mode_seg_one_point':     ('map_to_max', 'map_to_min', 'map_to_mean'),
        区间丢失情况，忽略，向下增加一个点，向上增加一个点，同时向下和向上增加点
          'mode_seg_non_point':     ('ignore', 'add_next_point', 'add_last_point', 'add_two_side'),
        区间端点是否共享
          'mode_seg_end_share':     ('no', 'yes'),

        标识星号（*）者是目前已经实现的，被认为是最重要的
        其余是默认第一选择值的，可以进一步研究的
"""


from collections import namedtuple
import scipy.stats as sts


# model type
MODEL_TYPE_PLT = 'plt'      # piecewise linear transform
MODEL_TYPE_SST = 'ppt'      # standard score transform
MODEL_TYPE = {MODEL_TYPE_PLT, MODEL_TYPE_SST}


# models parameters: grade score ratios, segments
CONST_ZHEJIANG_RATIO = (1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1)
CONST_ZHEJIANG_SEGMENT = ((100-i*3, 100-i*3) for i in range(21))
CONST_SHANGHAI_RATIO = (5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5)
CONST_SHANGHAI_SEGMENT = ((70-i*3, 70-i*3) for i in range(11))
CONST_BEIJING_RATIO = (1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1)
CONST_BEIJING_SEGMENT = ((100-i*3, 100-i*3) for i in range(21))
CONST_TIANJIN_RATIO = (2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1)
CONST_TIANJIN_SEGMENT = ((100-i*3, 100-i*3) for i in range(21))


# ShanDong
# 8 levels, (3%, 7%, 16%, 24%, 24%, 16%, 7%, 3%)
# 8 segments: (100, 91), ..., (30, 21)
CONST_SHANDONG_RATIO = (3, 7, 16, 24, 24, 16, 7, 3)
CONST_SHANDONG_SEGMENT = ((100-i*10, 100-i*10-9) for i in range(8))


# GuangDong
#   predict: mean = 70.21, std = 20.95
CONST_GUANGDONG_RATIO = (17, 33, 33, 15, 2)
CONST_GUANGDONG_SEGMENT = ((100, 83), (82, 71), (70, 59), (58, 41), (40, 30))


# 7-ShengShi: JIANGSU, FUJIAN, HUNAN, HUBEI, CHONGQING, HEBEI, LIAONING
#   5 levels
#   ration=(15%、35%、35%、13%, 2%),
#   segment=(86～100、71～85、56～70、41～55、30～40)
#   predict: mean = 70.24, std = 21.76
#            mean = sum((x/100*sum(y)/2 for x,y in zip(SS7ratio,SS7segment)))
#            std = math.sqrt(sum(((sum(y)/2-mean)**2 for x,y in zip(SS7ratio,SS7segment)))/5)
CONST_SS7_RATIO = (15, 35, 35, 13, 2)
CONST_SS7_SEGMENT = ((100, 86), (85, 71), (70, 56), (55, 41), (40, 30))



# get ratio from norm distribution table, scipy.stats.norm.cdf
# for standard score start-end range(100-900, 60-300,...)
def get_ratio_from_norm_cdf(start, end, std_num=4, step=1):
    """
    set endpoint ratio from morm.cdf:
        start_point: seg[0] = (1 - cdf(-4))*100
         next_point: seg_ratio = cdf[i+1] - cdf[i],
          end_point: seg[-1] = 100 - sum(seg[:-1])      # ensure to sum==100
    """
    start_point, end_point, _mean = start, end, (start+end)/2
    _std = (_mean - start_point) / std_num
    norm_cdf = [sts.norm.cdf((v-_mean)/_std) for v in range(start_point, end_point + 1, step)]
    norm_table = [(norm_cdf[i] - norm_cdf[i-1])*100 if i > 0
                  else norm_cdf[i]*100
                  for i in range(len(norm_cdf))]
    norm_table[-1] = 100 - sum(norm_table[:-1])
    return tuple(norm_table)


# Hainan standard score(old national) parameters(range:100-900, ratio: norm:(std=100, mean=500))
CONST_HAINAN_RATIO = get_ratio_from_norm_cdf(100, 900)
CONST_HAINAN_SEGMENT = ((s, s) for s in range(900, 100-1, -1))

# Hainan2 out_scope: 60-300 (mean=180, std=30)
#         ordinary method: transform each score individually
#         use norm cdf for each point, first set in 60-300, then pick ratio-score in raw segtable
CONST_HAINAN300_RATIO = get_ratio_from_norm_cdf(60, 300)
CONST_HAINAN300_SEGMENT = ((s, s) for s in range(300, 60 - 1, -1))

# Hainan3 out_scope 60-300,
#         use plt method to transform
#         set top fine proximate ratio to norm distribution
CONST_HAINAN3_RATIO = (0.14, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.14)
CONST_HAINAN3_SEGMENT = ((x, x - 30 + 1 if x > 90 else x - 30) for x in range(300, 90 - 1, -30))

# Hainan4 using plt for 60-300, use plt method to transform
#         set secondary proximate ratio
CONST_HAINAN4_RATIO = (0.2, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.2)
CONST_HAINAN4_SEGMENT = ((x, x - 30 + 1 if x > 90 else x - 30) for x in range(300, 90 - 1, -30))

# Hainan5 using plt for 60-300, use plt method to transform
#         set suitable ratio
CONST_HAINAN5_RATIO = (1, 2, 14, 33, 33, 14, 2, 1)
CONST_HAINAN5_SEGMENT = ((x, x - 30 + 1 if x > 90 else x - 30) for x in range(300, 90 - 1, -30))


ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'seg', 'desc'])
MODELS_SETTING_DICT = {
    'zhejiang':     ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_ZHEJIANG_RATIO),
                                tuple(CONST_ZHEJIANG_SEGMENT),
                                'piecewise linear transform model'),
    'shanghai':     ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_SHANGHAI_RATIO),
                                tuple(CONST_SHANGHAI_SEGMENT),
                                'piecewise linear transform model'),
    'beijing':      ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_BEIJING_RATIO),
                                tuple(CONST_BEIJING_SEGMENT),
                                'piecewise linear transform model'),
    'tianjin':      ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_TIANJIN_RATIO),
                                tuple(CONST_TIANJIN_SEGMENT),
                                'piecewise linear transform model'),
    'shandong':     ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_SHANDONG_RATIO),
                                tuple(CONST_SHANDONG_SEGMENT),
                                'piecewise linear transform model'),
    'guangdong':    ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_GUANGDONG_RATIO),
                                tuple(CONST_GUANGDONG_SEGMENT),
                                'piecewise linear transform model'),
    'ss7':          ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_SS7_RATIO),
                                tuple(CONST_SS7_SEGMENT),
                                'piecewise linear transform model'),
    'hn900':        ModelFields(MODEL_TYPE_SST,
                                tuple(CONST_HAINAN_RATIO),
                                tuple(CONST_HAINAN_SEGMENT),
                                'standard score model'),
    'hn300':        ModelFields(MODEL_TYPE_SST,
                                tuple(CONST_HAINAN300_RATIO),
                                tuple(CONST_HAINAN300_SEGMENT),
                                'standard score model'),
    'hn300plt1':    ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_HAINAN3_RATIO),
                                tuple(CONST_HAINAN3_SEGMENT),
                                'piecewise linear transform model'),
    'hn300plt2':    ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_HAINAN4_RATIO),
                                tuple(CONST_HAINAN4_SEGMENT),
                                'piecewise linear transform model'),
    'hn300plt3':    ModelFields(MODEL_TYPE_PLT,
                                tuple(CONST_HAINAN5_RATIO),
                                tuple(CONST_HAINAN5_SEGMENT),
                                'piecewise linear transform model with ratio-segment')
    }

# choice_count = 4 * 2 * 2 * 2 * 2 * 3 * 4 * 2 * 2 * 3 * 2,  18432
# key choice = 4 * 2 * 2 * 3 * 4    ## prox, cumu, sort, one_point, non_point
MODEL_STRATEGIES_DICT = {
    'mode_ratio_prox':          ('upper_min', 'lower_max', 'near_max', 'near_min'),
    'mode_ratio_cumu':          ('yes', 'no'),
    'mode_score_sort':          ('ascending', 'descending'),
    'mode_score_full_to_max':   ('ignore', 'yes'),    # full raw score to max out score, not only empty, also for ratio
    'mode_score_high_to_max':   ('ignore', 'yes'),    # max raw score to max out score
    'mode_score_zero_to_min':   ('ignore', 'yes'),    # min raw score to min out score, to this result if sort is 'a'
    'mode_seg_one_point':       ('map_to_max', 'map_to_min', 'map_to_mean'),
    'mode_seg_non_point':       ('ignore', 'add_next_point', 'add_last_point', 'add_two_side'),
    'mode_seg_2nd_point':       ('minus_one', 'jump_empty_point'),     # how to get second endpoint for each segment
    'mode_score_empty':         ('ignore', 'map_to_up', 'map_to_low'),    # ** consider to deprecated
    'mode_seg_end_share':       ('no', 'yes'),
    }
