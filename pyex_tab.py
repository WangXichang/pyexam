# -*- coding:utf-8 -*- 

from texttable import Texttable
import numpy as np
import pandas as pd
import pyex_lib as pl

def df_to_table(dataframe,
                title='new table',
                pagelines=30,
                writfile = '2015_2017score_stats.txt'):
    result_string = ''
    newtitle = ''
    pagewidth = 0
    pagenum = 0
    for linenum in range(pagelines, dataframe.count()[0]+pagelines, pagelines):
        print(linenum, dataframe.count()[0])
        newpage = maketab(dataframe.iloc[pagenum*pagelines:linenum]) + '\f\n\n'
        if linenum == pagelines:
            pagewidth = max([len(s) for s in newpage.split('\n')])
            newtitle = ' ' * int((pagewidth - len(title))/2) + title if pagewidth > len(title) else title
            title = newtitle
        pagenostr = ' ' * int((pagewidth - 8)/2) + '--' + pl.int_str(pagenum+1, 3) + '--'
        result_string = result_string +  newtitle + '\n' + newpage + '\n' + pagenostr + '\n'
        pagenum += 1
    if len(writfile) > 0:
        fh = open(writfile, 'w')
        fh.write(result_string)
        fh.close()
    return result_string


def maketab(df, *args, **kwargs):
    result_txt = tab_text(df, *args, **kwargs)
    # print(result_txt)
    return result_txt


# -----------------------new texttable for this package
# have modified texttable for produce normal line with chinese char
# texttable function --
# def len(iterable):
#   ...
#   before modifying:
#   w = unicodedata.east_asian_width
#   return sum([w(c) in 'WF' and 2 or 0 if unicodedata.combining(c) else 1 for c in unicode_data])
#   after modifying:
#   return sum([2 if uchar >= u'\u4e00' and uchar <= u'\u9fa5' else 1 for uchar in unicode_data])
# -------------------------improve texttable notes end
# columnsFormat = [[columnsNo,Width,dtype,,align],...]
#   style:
#           't',  # text
#           'f',  # float (decimal)
#           'e',  # float (exponent)
#           'i',  # integer
#           'a'])  # automatic
#   Api:    listTable
# --------------------------notes for parameters format
def tab_text(df, colwidth=None, colnames=None, vline=True):
    # hline=None, columnsformat = None):
    if type(df) is not pd.DataFrame:
        print('Warning:\n', type(df), '\n', df)
        print('input data is not DataFrame!')
        return
    colnum = df.columns.__len__()
    # rownum = df.__len__()
    table = Texttable()
    if not vline:
        table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l"] * colnum)
    table.set_cols_valign(["m"] * colnum)
    table.set_chars(["-", "|", "+", "="])
    table.set_cols_dtype(['t'] * colnum)
    if not colwidth:
        colwidth = {}
    elif type(colwidth) != dict:
        print('colWidth is not dict type!')
        return
    defaultwidth = [10] * colnum
    if len(colwidth) > 0:
        for k in colwidth:
            if (type(k) == int) & (k in range(1, colnum+1)):
                defaultwidth[k-1] = colwidth[k]
            else:
                print('colwidth is error!')
    table.set_cols_width(defaultwidth)
    if colnames:
        headnames = colnames
    else:
        headnames = [s for s in df.columns]
    rowall = [headnames] + \
             [list(df.values[i]) for i in range(len(df))]
    table.add_rows(rowall)
    rr = table.draw()
    return rr


# chinese char processing function
def reduce_blank_for_str(ctr):
    lstr = ctr.split(sep='\n')
    rstr = ''
    for s in lstr:
        if s == '':
            continue
        lls = s.split(sep='|')
        nlls = [ss if _count_chinese_char(ss) == 0
                else ss[0:len(ss) - _count_chinese_char(ss)]
                for ss in lls]
        rstr = rstr + ''.join(['|'+r if r != '' else '' for r in nlls]+['|'])+'\n'
    return rstr


def _is_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def _count_chinese_char(ustr):
    return sum([1 if _is_chinese_char(c) else 0 for c in ustr])


def str_width(ustr):
    return len(ustr) + _count_chinese_char(ustr)


# create data example
def test_df_with_cc():
    r = pd.DataFrame({"family": ["王莽族", "张春秋大家族人很多", "郭子仪"],
                      "level": [1, 2, 3],
                      "name": ["Robit欧洲人", "Smith美国人", "希腊Garisson"]
                      })
    return r


def expsr():
    r = pd.Series(['Robit', 'Smith', 'Family'], index=range(3), name='name')
    return r


def exp_normal_df(colnum=1, size=10, delta=1, mu=0):
    columnslist = ['v%d' % i for i in range(colnum)]
    vallist = {columnslist[i]: np.random.randn(size)*delta + mu for i in range(colnum)}
    return pd.DataFrame(vallist)


def test_table_for_cc():
    # rowAll= [list(df.columns.values)] \
    #    +[list(df.ix[ri]) for ri in range(len(df))]
    table = Texttable()
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])
    old = 1
    if old == 1:
        rowlist = [["Name", "Age", "Nickname"],
                   ["Mr\n胡大山\n胡树林", 32, "Xav'"],
                   ["Mr\\n陈清器\\n刘为级", 1, "Baby"],
                   ["Mme\\nLouise\\n德里市", 28, "Lou\\n\\nLoue"]]
        table.add_rows(rowlist)
    else:
        table.add_rows("")
    print(table.draw() + "\n")
    return
