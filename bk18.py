
def find(zy):
    fxx = 0
    if fxx:
        low = 115000
        high = 150000
        filterlist = ['校企']
        kl = 'lk'
        df = zy.findxx(low=low, high=high, filterlist=filterlist, kl=kl, align={'xx': 'l', 'lkjh': 'r'})
        pass
    fzy = 0
    if fzy:
        low = 40000
        high = 180000
        zyfilter = ['法学']
        xxfilter = ['山东']
        zy.findzy(lowpos=low, highpos=high, zyfilterlist=zyfilter, xxfilterlist=xxfilter)
        pass
    fsx = 10
    if fsx:
        # filterlist = ['北京电影', '西南大学', '广州大学', '天津工业大学', '青岛大学', '山东师范大学']
        # filterlist = ['电子科技大学', '吉林大学', '哈尔滨工业']
        filterlist = ['校企']
        kl = 'wk'
        df = zy.somexx(xxsubstr=filterlist, kl=kl)
        pass

    return


def ys_zf(wh=100, zy=200):
    return wh*0.3 + zy*750/300*0.7


def ty_zf(wh=100, zy=100):
    return wh*0.3 + zy*750/100*0.7
