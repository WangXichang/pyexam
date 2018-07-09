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


def make_mpage(df, title='', pagelines=30, align=None, fold=0):
    if align is None:
        align = dict()
    gridnum = len(df.columns)
    result = ''
    ptext = make_table(df=df, title=title, align=align)
    plist = ptext.split('\n')
    plen = len(plist)

    # retrieving table head and gapline
    hline = 0
    head = ''
    gapline = None
    pagewid = 0
    i = 0
    while True:
        # set subtitle to center
        if ('+' not in plist[i]) & (plist[i].count('|') == gridnum + 1):
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
        if plist[i].count('+') == gridnum + 1:
            hline = hline + 1
            if gapline is None:
                pagewid = len(plist[i])
                gapline = plist[i] + '\n'
        # retrieving head finished
        if hline == 2:
            break

    # construct fold head
    mhead = ''
    if fold > 0:
        hlist = head.split('\n')
        mhlist = []
        for hs in hlist:
            mhlist.append((hs + '\t') * fold + hs + '\n')

    # construct pages
    result = ''
    mpagetext = ''
    pagetext = ''
    textline = 0
    pageno = 0
    # page_fold_line = pagelines * (fold + 1)
    cur_fold = fold
    for j in range(i+1, plen):
        pagetext += plist[j] + '\n'
        textline += 1
        if textline == pagelines:
            if cur_fold == fold:
                mpagetext = pagetext
                pagetext = ''
                cur_fold -= 1
            else:
                mpagetext = concat_page(mpagetext, pagetext)
            textline = 0
        # find gapline and the end of table
        if plist[j].count('+') == gridnum + 1:
            break

    return result


def concat_page(mp, pt):
    mpl = mp.split('\n')
    npl = pt.split('\n')
    for j in range(len(mpl)):
        mpl[j] = mpl[j] + '\t' + npl[j] + '\n'
    return ''.join(mpl)
