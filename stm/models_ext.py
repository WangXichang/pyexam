# coding: utf-8

"""
    本模块用于增加新的模型，或修改自定义模型。

    定义模型
    （1）模型
    模型由一个名称name和带有4个字段的tuple组成。
    每个模型都是模型库的一个键值对成员，通过Models_ext设置。
    设置在Models_ext中的模型，在运行main.run时， 可以通过设置reload=True，使系统自动加入到Models中。
    MODEL = {name: ModelFields}
    ModelFields: {
                  'type':       str := 'plt' or 'ppt',
                  'ratio':      list or tuple:= percent value for each section,
                                100 percent: 0-100,
                                descending: from big section to small, corresponding to the position of section
                                i.e. (1, 2, ..., 1), sum(ratio)==100
                  'section':    list or tuple:= output score section,
                                descending: from big score to min score
                                i.e. [(p11, p12), ...(pn1, pn2)], pi1 < pi2, and, pji < pk1 for k>j
                  'desc':       str := description for model
                  }
    （2）参数描述
     1）name
    模型名称，尽量避免与 modelsetin.Models 中的已有名称重复, 否则会覆盖内置的模型;
    可以通过查看modelsetin 或 使用 name in modelsetin.Models.keys() 进行检查。
    使用main.run(..., reload=True)时会检查提示。
     2）type
    'plt',    # 分段线性转换 piecewise linear transform
    'ppt'     # 逐点转换 piecewise point transform
    
     3) ratio
    每个分数区间的考生人数占比，
    每个比例对应与section中的区间：ratio[i] --> section[i]
    比例数值的个数需要与区间数相同： len(ratio) == len(section)
    比例和为100：sum(ratio) == 100
    在plt中，用于定义原始分数划分区间，
    在ppt中，用于定义转换分数的分值点比例
    在pgt中，用第一个比例给出高分区，通过计算高分区均值给出第一个区间的第二端点。其余比例不使用。

     4）section
    转换分数区间，个数与比例个数相同。
    端点顺序从大到小，即从高分到低分
    在plt中，用于将原始分数的每个区间映射到section区间，端点根据标准确定。
    在ppt中，用于将原始分数的每个分值映射到section端点，ppt的区间端点值是相等的，为本分值。
    在pgt中，用于将原始分数等级区间值映射到section端点, pgt的区间端点值是相同的，为本等级值。

    注1：如果在本模块中修改增加了模型，使用main.run时需要设置reload=True，重新载入Models_ext
    注2：在使用 main.run 或 main.run_model 调用新定义的模型前，可以使用modelutil.check_model检查是否存在设置问题。
"""


from collections import namedtuple


ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'section', 'desc'])
Models = {
    'hn300plt1':    ModelFields(
                                'plt',
                                (0.14, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.14),
                                tuple((x, x-30+1) if x > 90 else (x, x-30) for x in range(300, 60, -30)),
                                # ((300, 271), (270, 241), ... , (120, 91), (90, 60)),
                                'piecewise linear transform model'
                                ),
    'hn300plt2':    ModelFields(
                                'plt',
                                (0.2, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.2),
                                tuple((x, x - 30 + 1) if x > 90 else (x, x - 30) for x in range(300, 60, -30)),
                                # ((300, 271), (270, 241), ... , (120, 91), (90, 60)),
                                'piecewise linear transform model'
                                ),
    'hn300plt3':    ModelFields(
                                'plt',
                                (1, 2, 14, 33, 33, 14, 2, 1),
                                tuple((x, x - 30 + 1) if x > 90 else (x, x - 30) for x in range(300, 60, -30)),
                                # ((300, 271), (270, 241), ... , (120, 91), (90, 60)),
                                'piecewise linear transform model with ratio-segment'
                                ),
    'hn300plt4':    ModelFields(
                                'plt',
                                (3, 14, 33, 33, 14, 3),
                                tuple((x, x - 40 + 1) if x > 100 else (x, x - 40) for x in range(300, 60, -40)),
                                # ((300, 261), (270, 221), ... , (140, 101), (100, 60)),
                                'piecewise linear transform model with ratio-segment'
                                ),
    'exp':    ModelFields(
                          'plt',
                          tuple(1/15 * 100 for _ in range(15)),
                          tuple((i+1, i+1) for i in range(15)),
                          'section wise linear transform model'
                          ),
    }
