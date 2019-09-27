# -*- utf8 -*-

import fractions as fra


def test(df=None, f='', r=0.3):
    percent_list = [0 for j in range(100)]
    for ri, row in df.iterrows():
        _fr = row[f+'_fr']
        for j in range(100):
            if fra.Fraction(j/100) < _fr <= fra.Fraction((j+1)/100):
                percent_list[j] += 1

    return percent_list
