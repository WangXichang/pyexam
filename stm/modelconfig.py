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
        CONST_SHANDONG_SECTION = [(21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
        CONST_SS7_RATIO = [2, 13, 35, 35, 15]
        CONST_SS7_SECTION = [(30, 40), (41, 55), (56, 70), (71, 85), (86, 100)]

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
import numpy as np
import matplotlib.pyplot as plot

# model type
MODEL_TYPE_PLT = 'plt'      # piecewise linear transform
MODEL_TYPE_PPT = 'ppt'      # standard score transform
# MODEL_TYPE = {MODEL_TYPE_PLT, MODEL_TYPE_PPT}

hn900model = ModelTools.get_section_pdf(100, 900, 800, 4, True, 'ppt', 100, 'desceding')
hn300model = ModelTools.get_section_pdf(60, 300, 240, 4, True, 'ppt', 100, 'descending')
zscoremodel = ModelTools.get_section_pdf(-4, 4, 800, 4, True, 'ppt', 100, 'd')
tscoremodel = ModelTools.get_section_pdf(100, 900, 800, 4, True, 'ppt', 100, 'd')

ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'section', 'desc'])
Models = {
    'zhejiang':     ModelFields(MODEL_TYPE_PLT,
                                (1, 2, 3, 4, 5, 6, 7, 8, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1),
                                tuple((x, x) for x in range(100, 39, -3)),
                                'piecewise linear transform model'),
    'shanghai':     ModelFields(MODEL_TYPE_PLT,
                                (5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5),
                                tuple((x, x) for x in range(70, 39, -3)),
                                'piecewise linear transform model'),
    'beijing':      ModelFields(MODEL_TYPE_PLT,
                                (1, 2, 3, 4, 5, 7, 8, 9, 8, 8, 7, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1),
                                tuple((100-i*3, 100-i*3) for i in range(21)),
                                'piecewise linear transform model'),
    'tianjin':      ModelFields(MODEL_TYPE_PLT,
                                (2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 1, 1, 1),
                                tuple((100-i*3, 100-i*3) for i in range(21)),
                                'piecewise linear transform model'),
    'shandong':     ModelFields(MODEL_TYPE_PLT,
                                (3, 7, 16, 24, 24, 16, 7, 3),
                                tuple((100-i*10, 100-i*10-9) for i in range(8)),
                                'piecewise linear transform model'),
    'guangdong':    ModelFields(MODEL_TYPE_PLT,
                                (17, 33, 33, 15, 2),
                                ((100, 83), (82, 71), (70, 59), (58, 41), (40, 30)),
                                'piecewise linear transform model'),
    'ss7':          ModelFields(MODEL_TYPE_PLT,
                                (15, 35, 35, 13, 2),
                                ((100, 86), (85, 71), (70, 56), (55, 41), (40, 30)),
                                'piecewise linear transform model'),
    'hn900':        ModelFields(MODEL_TYPE_PPT,
                                hn900model.pdf,
                                hn900model.section,
                                'standard score model: piecewise point transform'),
    'hn300':        ModelFields(MODEL_TYPE_PPT,
                                hn300model.pdf,
                                hn300model.section,
                                'standard score model: piecewise point transform'),
    'hn300plt1':    ModelFields(
                        MODEL_TYPE_PLT,
                        (0.14, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.14),
                        ((300, 271), (270, 241), (240, 211), (210, 181), (180, 151), (150, 121), (120, 91), (90, 60)),
                        'piecewise linear transform model'),
    'hn300plt2':    ModelFields(
                        MODEL_TYPE_PLT,
                        (0.2, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.2),
                        ((300, 271), (270, 241), (240, 211), (210, 181), (180, 151), (150, 121), (120, 91), (90, 60)),
                        'piecewise linear transform model'),
    'hn300plt3':    ModelFields(
                        MODEL_TYPE_PLT,
                        (1, 2, 14, 33, 33, 14, 2, 1),
                        ((300, 271), (270, 241), (240, 211), (210, 181), (180, 151), (150, 121), (120, 91), (90, 60)),
                        'piecewise linear transform model with ratio-segment'),
    'zscore':       ModelFields(MODEL_TYPE_PPT,
                                zscoremodel.pdf,
                                zscoremodel.section,
                                'piecewise linear transform model with ratio-segment'),
    'tscore':       ModelFields(MODEL_TYPE_PPT,
                                tscoremodel.pdf,
                                tscoremodel.section,
                                'piecewise linear transform model with ratio-segment'),
    'tai':          ModelFields(MODEL_TYPE_PLT,
                                (),
                                (),
                                'piecewise linear transform model with ratio-segment'),
    }

# choice_space = 4 * 2 * 2 * 2 * 2 * 3 * 4 * 2 * 2 * 3 * 2,  18432
# real used choice = 4 * 2 * 2 * 3 * 2 * 3 = 96    ## prox, cumu, sort, one_point, non_point
Strategies = {
    'mode_ratio_prox':              ('upper_min', 'lower_max', 'near_max', 'near_min'),
    'mode_ratio_cumu':              ('yes', 'no'),
    'mode_sort_order':              ('ascending', 'descending'),
    'mode_section_point_first':     ('real', 'defined'),      # first point of first section
    'mode_section_point_start':     ('step', 'share'),      # first point except first section
    'mode_section_point_last':      ('real', 'defined'),      # useful to type--ppt
    'mode_section_degraded':        ('map_to_max', 'map_to_min', 'map_to_mean'),
    'mode_section_lost':            ('ignore', 'next_one_point', 'next_two_point'),
    }

# to add in future
# choice_space: 2 * 2 * 4 * 3 = 48
# MODEL_STRATEGIES_RESERVE_DICT = {
# 'mode_ppt_score_max': ('map_by_real_ratio', 'map_to_max'),  # for standard score transform: type=='ppt'
# 'mode_ppt_score_min': ('map_by_real_ratio', 'map_to_min'),  # for standard score transform: type=='ppt'
#                                                                              # first point by mode_section_min/max
#     'mode_section_lost':                ('ignore', 'add_next_point', 'add_last_point', 'add_two_side'),
#     'mode_section_min':                 ('real_min', 'defined_min'),
#     'mode_section_max':                 ('real_max', 'defined_max'),
#       'mode_score_empty': ('use', 'jump'),  # ** consider to deprecated, processed in other strategies
#       'mode_score_rmin_to_min': ('ignore', 'yes'),  # real raw score min value to out score min value,
#                                                     # case: sort by 'a', standard score mode
#       'mode_score_rmax_to_max': ('ignore', 'yes'),  # real raw score max value to out score max value,
#                                                     # case: top ratio large, sort by 'a', standard score mode
# }


class ModelTools:

    @classmethod
    def get_section_pdf(cls,
                        start=21,
                        end=100,
                        section_num=8,
                        std_num=2.6,
                        add_cutoff=True,
                        model_type='plt',
                        ratio_coeff=1,      #1, or 100
                        sort_order='d',
                        ):
        """
        # get pdf, cdf, cutoff_err form section end points,
        # set first and end seg to tail ratio from norm table
        # can be used to test std from seg ratio table
        # for example,
        #   get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.9508)
        #   [0.03000, 0.07513, 0.16036, 0.234265, ..., 0.03000],
        #   get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.6065)
        #   [0.02729, 0.07272, 0.16083, 0.23916, ..., 0.02729],
        #   it means that std==15.95 is fitting ratio 0.03,0.07 in the table

        :param start:   start value
        :param end:     end value
        :param section_num: section number
        :param std_num:     length from 0 to max equal to std_num*std, i.e. std = (end-start)/2/std_num
        :param add_cutoff: bool, if adding cutoff cdf() at edge point
                           i.e. cdf(-std_num), cdf(-4) = 3.167124183311986e-05, cdf(-2.5098)=0.029894254950869625
        :param model_type: str, 'plt' or 'ppt'
        :return: namedtuple('result', ('section':((),...), 'pdf': (), 'cdf': (), 'cutoff': float, 'add_cutoff': bool))
        """
        _mean, _std = (end+start)/2, (end-start)/2/std_num
        section_point_list = np.linspace(start, end, section_num+1)
        cutoff = sts.norm.cdf((start-_mean)/_std)
        pdf_table = [0]
        cdf_table = [0]
        last_pos = (start-_mean)/_std
        _cdf = 0
        for i, pos in enumerate(section_point_list[1:]):
            _zvalue = (pos-_mean)/_std
            this_section_pdf = sts.norm.cdf(_zvalue)-sts.norm.cdf(last_pos)
            if (i == 0) and add_cutoff:
                this_section_pdf += cutoff
            pdf_table.append(this_section_pdf)
            cdf_table.append(this_section_pdf + _cdf)
            last_pos = _zvalue
            _cdf += this_section_pdf
        if add_cutoff:
            pdf_table[-1] += cutoff
            cdf_table[-1] = 1
        if model_type == 'plt':
            section_list = [(x, y) if i == 0 else (x+1, y)
                            for i, (x, y) in enumerate(zip(section_point_list[:-1], section_point_list[1:]))]
        else:
            section_list = [(x, x) for x in section_point_list]
        if ratio_coeff != 1:
            pdf_table = [x*ratio_coeff for x in pdf_table]
        if sort_order in ['d', 'descending']:
            section_list = sorted(section_list, key=(lambda x: -x[0]))
        result = namedtuple('Result', ('section', 'pdf', 'cdf', 'point', 'cutoff', 'add_cutoff'))
        r = result(tuple(section_list),
                   tuple(pdf_table),
                   tuple(cdf_table),
                   section_point_list,
                   cutoff,
                   add_cutoff)
        return r

    @classmethod
    def plot_models(cls, font_size=12, hainan='900'):
        _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'ss7', 'hn900']
        if hainan == '300':
            _names.remove('hn900')
            _names.append('hn300')
        elif hainan is None:
            _names.remove('hn900')
        ms_dict = dict()
        for _name in _names:
            ms_dict.update({_name: ModelTools.get_model_describe(name=_name)})

        plot.figure('New Gaokao Score Models: name(mean, std, skewness)')
        plot.rcParams.update({'font.size': font_size})
        for i, k in enumerate(_names):
            plot.subplot(240+i+1)
            _wid = 2
            if k in ['shanghai']:
                x_data = range(40, 71, 3)
            elif k in ['zhejiang', 'beijing', 'tianjin']:
                x_data = range(40, 101, 3)
            elif k in ['shandong']:
                x_data = [x for x in range(26, 100, 10)]
                _wid = 8
            elif k in ['guangdong']:
                x_data = [np.mean(x) for x in Models[k].section][::-1]
                _wid = 10
            elif k in ['ss7']:
                x_data = [np.mean(x) for x in Models[k].section][::-1]
                _wid = 10
            elif k in ['hn900']:
                x_data = [x for x in range(100, 901)]
                _wid = 1
            elif k in ['hn300']:
                x_data = [x for x in range(60, 301)]
                _wid = 1
            else:
                raise ValueError(k)
            plot.bar(x_data, Models[k].ratio[::-1], width=_wid)
            plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))

    @classmethod
    def add_model(cls, model_type='plt', name=None, ratio_list=None, section_list=None, desc=''):
        if model_type not in ['ppt', 'plt']:
            print('error model type={}, valid type:{}'.format(model_type, ['ppt', 'plt']))
            return
        if name in Models:
            print('name existed in current models_dict!')
            return
        if len(ratio_list) != len(section_list):
            print('ratio is not same as segment !')
            return
        for s in section_list:
            if len(s) > 2:
                print('segment is not 2 endpoints: {}-{}'.format(s[0], s[1]))
                return
            if s[0] < s[1]:
                print('the order is from large to small: {}-{}'.format(s[0], s[1]))
                return
        if not all([s1 >= s2 for s1, s2 in zip(section_list[:-1], section_list[1:])]):
            print('section endpoints order is not from large to small!')
            return
        Models.update({name: ModelFields(
                        model_type,
                        ratio_list,
                        section_list,
                        desc)})

    @classmethod
    def show_models(cls):
        for k in Models:
            v = Models[k]
            print('{:<20s} {},  {} '.format(k, v.type, v.desc))
            print('{:<20s} {}'.format(' ', v.ratio))
            print('{:<20s} {}'.format('', v.section))

    @classmethod
    def get_model_describe(cls, name='shandong'):
        __ratio = Models[name].ratio
        __section = Models[name].section
        if name == 'hn900':
            __mean, __std, __skewness = 500, 100, 0
        elif name == 'hn300':
            __mean, __std, __skewness = 180, 30, 0
        else:
            samples = []
            [samples.extend([np.mean(s)]*int(__ratio[i])) for i, s in enumerate(__section)]
            __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))
        return __mean, __std, __skewness
