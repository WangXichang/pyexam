# coding: utf-8

"""
    定义模型
    （1）模型
    模型由一个名称name和带有4个字段的tuple组成。
    每个模型都是模型库的一个键值对成员，通过Models_ext设置。
    设置在Models_ext中的模型，在运行main.run时， 可以通过设置reload=True，使系统自动加入到Models中。
    MODEL = {name: ModelFields}
    ModelFields: {
                  'type':       str := 'plt' or 'ppt',
                  'ratio':      list or tuple:= percent value for each section,
                                i.e. (1, 2, ..., 1), sum==100
                  'section':    list or tuple:= output score section,
                                i.e. [(p11, p12), ...(pn1, pn2)], pi1 > pi2
                  'desc':       str := description for model
                  }
    （2）参数描述
     1）name
    模型名称，不能与已有名称（modelsetin.Models.keys()）重复, 否则会覆盖内置的模型;
    
     2）type
    'plt',    # 分段线性转换 piecewise linear transform
    'ppt'     # 逐点转换 piecewise point transform
    
     3) ratio
    每个分数区间的比例，个数需要与区间数相同
    在plt中，用于定义原始分数划分区间，
    在ppt中，用于定义转换分数的分值点比例
    在pgt中，用第一个比例给出高分区，通过计算高分区均值给出第一个区间的第二端点。其余比例不使用。

     4）section
    转换分数区间，个数与比例个数相同。
    在plt中，用于将原始分数的每个区间映射到section区间，端点根据标准确定。
    在ppt中，用于将原始分数的每个分值映射到section端点，ppt的区间端点值是相等的，为本分值。
    在pgt中，用于将原始分数等级区间值映射到section端点, pgt的区间端点值是相同的，为本等级值。
    
"""


from collections import namedtuple


ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'section', 'desc'])
Models_ext = {
    'exp':    ModelFields(
                          'plt',
                          tuple(1/15 * 100 for _ in range(15)),
                          tuple((i+1, i+1) for i in range(15)),
                          'section wise linear transform model'
                          ),
    }
# note: must check the model by modelutil.check_model
#       then can use the model in main.run, main.run_model
