# -*- utf8 -*-


import fractions as fra
import functools as ft


def test(df=None, f='', r=0.3):
    _r = fra.Fraction(r).limit_denominator(10000)
    print(_r)
    # for x, y in zip(list(df['seg']), list(df[f+'_fr'])):
    #     if fra.Fraction(y) <= _r:
    #         print(x, y)
    for ri, row in df.iterrows():
        _fr = row[f+'_fr']
        if _r <= _fr:
            print(_r, _fr, _r >= _fr)
            return r, row['seg'], row[f+'_percent'], _fr


def test_reduce():
    _list = [1,2,3,4,5]
    _loc = 2.5
    f = lambda x,y: (x, y) if y >= _loc else y
    return


def _seek(data1=(), data2=(), loc=None):
    for i, (score, seg) in enumerate(zip(data1, data2)):
        if score >= loc and i > 0:
            return (data1[i-1], data2[i-1]), (score, seg)


def round45r(number, digits=0):
    int_len = len(str(int(abs(number))))
    signal_ = 1 if number >= 0 else -1
    err_place = 16 - int_len - 1
    if err_place > 0:
        err_ = 10**-err_place
        return round(number + err_*signal_, digits) + err_ * signal_
    else:
        raise NotImplemented
