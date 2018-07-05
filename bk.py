
def find(zy):
    fxx = 10
    fzy = 0
    fsx = 0
    if fxx:
        low = 10000
        high = 20000
        filterlist = ['']
        df = zy.findxx(low=low, high=high, filterlist=filterlist, kl='lk', align={'xx': 'l', 'lkjh': 'r'})
        pass
    if fzy:
        low = 40000
        high = 80000
        zyfilter = ['临床']
        xxfilter = ['']
        zy.findzy(lowpos=low, highpos=high, zyfilterlist=zyfilter, xxfilterlist=xxfilter)
        pass
    if fsx:
        filterlist = ['山东财经大学']
        kl = 'lk'
        df = zy.somexx(xxsubstr=filterlist, kl=kl)
        pass

    return


def ys_zf(wh=100, zy=200):
    return wh*0.3 + zy*750/300*0.7


def ty_zf(wh=100, zy=100):
    return wh*0.3 + zy*750/100*0.7
