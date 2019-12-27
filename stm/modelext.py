# coding: utf-8


from collections import namedtuple


"""
    模型定义新的模型
    （1）模型定义
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
    模型名称，不能与已有名称（modelsetin.Models）重复;
    
      2）type
    'plt',    # 分段线性转换 piecewise linear transform
    'ppt'     # 逐点转换 piecewise point transform
    
      3) ratio
    每个分数区间的比例，
    在plt中用于定义原始分数划分区间，
    在ppt中用于定义转换分数的分值点比例
    
      4）section
    转换分数区间
    在plt中，用于将原始分数的每个区间映射到section中的区间
    在ppt中，用于将原始分数的每个分值映射到section中的值点，ppt中的区间端点是相等的。
    
"""


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
