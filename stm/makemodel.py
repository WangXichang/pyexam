# coding: utf-8


from _collections import namedtuple


ModelFields = namedtuple('ModelFields', ['type', 'ratio', 'section', 'desc'])
Models_ext={
    'exp':          ModelFields(
                                'plt',
                                [1/15 *100 for _ in range(15)],     # only first==1 is useful
                                tuple((i+1, i+1) for i in range(15)),
                                'piecewise linear transform model with ratio-section'
                                ),
    }
