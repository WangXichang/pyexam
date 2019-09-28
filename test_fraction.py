# -*- utf8 -*-


import fractions as fra
import functools as ft


def test(df=None, f='', r=0.3):
    percent_list = [0 for j in range(100)]
    for ri, row in df.iterrows():
        _fr = row[f+'_fr']
        for j in range(100):
            if fra.Fraction(j/100) < _fr <= fra.Fraction((j+1)/100):
                percent_list[j] += 1
    return percent_list


def test_reduce():
    _list = [1,2,3,4,5]
    _loc = 2.5
    f = lambda x,y: (x, y) if y >= _loc else y
    return


def _seek(data1=(), data2=(), loc=None):
    for i, (score, seg) in enumerate(zip(data1, data2)):
        if score >= loc and i > 0:
            return (data1[i-1], data2[i-1]), (score, seg)
