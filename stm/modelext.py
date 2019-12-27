# coding: utf-8


from _collections import namedtuple

"""
    模型定义

    分数转换方式：
    'plt',    # 分段线性转换 piecewise linear transform
    'ppt'     # 逐点转换 piecewise point transform

    模型参数
    MODEL = {name: ModelFields}
    ModelFields: {
                  'type': str := 'plt' or 'ppt',
                  'ratio': list or tuple:= percent value for each sectio, 
                           i.e. (1, 2, ..., 1), sum==100 
                  'section': list or tuple:= output score section, 
                             i.e. [(p11, p12), ...(pn1, pn2)], pi1 > pi2
                  'desc': str := description for model
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
