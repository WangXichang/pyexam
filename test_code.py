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
    # ( 1)                9 bit=4
    # ( 2)               99 bit=7
    # ( 3)              999 bit=10
    # ( 4)             9999 bit=14
    # ( 5)            99999 bit=17
    # ( 6)           999999 bit=20
    # ( 7)          9999999 bit=24
    # ( 8)         99999999 bit=27
    # ( 9)        999999999 bit=30
    # (10)       9999999999 bit=34
    # (11)      99999999999 bit=37
    # (12)     999999999999 bit=40
    # (13)    9999999999999 bit=44
    # (14)   99999999999999 bit=47
    # (15)  999999999999999 bit=50
    # (16) 9999999999999999 bit=54
    # ---------------------------------------------
    # format(ttc.round45r(999999999.145,2), '.20f')
    # '999999999.15000021457672119141'
    # note: valid digits is less than 52-int_bit_len
    #       err_ must be after round place
    #       int_bit + digits(10?) + 1 >= 52
    # ---------------------------------------------
    bit_list = [4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50]
    int_bit_len = int(number).bit_length()
    int_len = len(str(int(abs(number))))
    if int_len + abs(digits) > 16:
        raise ValueError
    signal_ = 1 if number >= 0 else -1
    err_place2 = 52 - int_bit_len - bit_list[int_len]
    if err_place2 > 1:
        err_ = signal_*2**-(52-int_bit_len)
        return round(number + err_, digits) + err_
    else:
        raise NotImplemented
