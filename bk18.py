
def find(zy):
    fxx = 10
    if fxx:
        low = 115000
        high = 150000
        filterlist = ['校企']
        kl = 'lk'
        df = zy.findxx(low=low, high=high, filterlist=filterlist, kl=kl, align={'xx': 'l', 'lkjh': 'r'})
        pass
    fzy = 10
    if fzy:
        low = 40000
        high = 180000
        zyfilter = ['法学']
        xxfilter = ['山东']
        zy.findzy(lowpos=low, highpos=high, zyfilterlist=zyfilter, xxfilterlist=xxfilter)
        pass
    fsx = 0
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
