# _*- coding: utf-8 -*-

from prettytable import PrettyTable as Ptt


def make_table(df, title='', align=None):
    align = {} if align is None else align
    x = Ptt()
    j = 0
    for f in df.columns:
        x.add_column(f, [x for x in df[f]])
        if (f in align):
            if (align[f] in ['l', 'c', 'r']):
                x.align[f] = align[f]
            elif df[f]._is_numeric_mixed_type:
                x.align[f] = 'r'
            elif df[f]._is_mixed_type:
                x.align[f] = 'l'
            else:
                x.align[f] = 'c'
        j = j + 1
    rs = x.get_string()
    return title.center(rs.index('\n')) + '\n' + rs


def make_page(df, title='', pagelines=30, align=None):
    align = {} if align is None else align
    gridnum = len(df.columns)
    result = ''
    ptext = make_table(df=df, title=title, align=align)
    plist = ptext.split('\n')
    # print(plist)
    plen = len(plist)
    hline = 0
    textline = 0
    head = ''
    gapline = None
    pagewid = 0
    pageno = 0
    for i in range(plen):
        if hline < 2:
            # set subtitle in center
            if ('+' not in plist[i]) & (plist[i].count('|') == gridnum + 1):
                sp = plist[i].split('|')
                newsp = []
                for x in sp:
                    if len(x.strip()) < len(x):
                        left_space = int((len(x) - len(x.strip()))/2)
                        newsp.append(' '*left_space + x.strip() + ' '*(len(x) - left_space-len(x.strip())))
                head += '|' + '|'.join(newsp) + '|\n'
            else:
                head += plist[i] + '\n'
        else:
            # not save first head in result
            if i < plen - 1:
                result += plist[i] + '\n'
        # find gapline and the end of head
        if plist[i].count('+') == gridnum + 1:
            hline = hline + 1
            if gapline is None:
                pagewid = len(plist[i])
                gapline = plist[i] + '\n'
            continue
        # add first head+gapline in result
        if (len(result) == 0) & (gapline is not None):
            result = head + gapline
        # start count content row number(textline)
        if hline == 2:
            textline += 1
        # seperate pages
        if (textline == pagelines) | (i == plen-2):
            pageno += 1
            pagenostr = ('--'+str(pageno)+'--').center(pagewid) + '\n\n'
            result += gapline + pagenostr + (head if i < plen-2 else '')
            textline = 0
    return result


def make_mpage(df, title='', page_line_num=30, align=None, fold=0):
    if align is None:
        align = dict()
    # columns_num = len(df.columns)
    ptext = make_table(df=df, title=title, align=align)
    plist = ptext.split('\n')
    plen = len(plist)

    # retrieving table head and gapline
    head, gapline,head_rows = get_head(plist)

    # construct pages
    result = ''
    mpagetext = ''
    pagetext = head
    pageline = 0
    pageno = 1
    cur_fold = fold
    for j in range(head_rows, plen):
        pagetext += plist[j] + '\n'
        pageline += 1
        # full page or end of table
        if (pageline == page_line_num) | ('+' in plist[j]):
            pagetext += (gapline if '+' not in plist[j] else '')\
                        + str(pageno).center(len(plist[j])) + '\n'
            if cur_fold == fold:
                mpagetext = pagetext
            else:
                if j < plen-1:
                    mpagetext = concat_page(mpagetext, pagetext)
            pagetext, pageline, pageno, cur_fold = head, 0, pageno+1, cur_fold-1
        if cur_fold < 0:
            result += mpagetext + '\n\f'
            mpagetext, pagetext, pageline, cur_fold = '', head, 0, fold
    return result


def get_head(plist):
    head, gapline = '', None
    hline, i, column_rownum = 0, 0, 0
    while hline < 2:
        # set subtitle to center
        if ('+' not in plist[i]) & (plist[i].count('|') > 1):
            # column_rownum = plist[i].count('|') - 1
            sp = plist[i].split('|')
            newsp = []
            for x in sp:
                if len(x.strip()) < len(x):
                    left_space = int((len(x) - len(x.strip())) / 2)
                    newsp.append(' ' * left_space + x.strip() + ' ' * (len(x) - left_space - len(x.strip())))
            head += '|' + '|'.join(newsp) + '|\n'
        else:
            head += plist[i] + '\n'
        # set gapline
        if plist[i].count('+') > 1:
            hline = hline + 1
            if gapline is None:
                gapline = plist[i] + '\n'
        i += 1

    return head, gapline, i


def concat_page(mp, pt):
    mpl = mp.split('\n')
    npl = pt.split('\n')
    for j in range(len(mpl)):
        if j < len(npl):
            mpl[j] = mpl[j] + '\t' + npl[j]
    return '\n'.join(mpl)
