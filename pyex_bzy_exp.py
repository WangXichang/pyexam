
def find(zy):
    fxx = 0
    if fxx:
        low = 115000
        high = 150000
        filterlist = ['校企']
        kl = 'lk'
        df = zy.findxx(low=low, high=high, filterlist=filterlist, kl=kl, align={'xx': 'l', 'lkjh': 'r'})
        pass
    fzy = 10
    if fzy:
        low = 76000
        high = 100000
        zyfilter = ['护理']
        xxfilter = ['']
        zy.findzy(lowpos=low, highpos=high, zyfilterlist=zyfilter, xxfilterlist=xxfilter)
        pass
    somexx = 1
    if somexx:
        # filterlist = ['北京电影', '西南大学', '广州大学', '天津工业大学', '青岛大学', '山东师范大学']
        # filterlist = ['电子科技大学', '吉林大学', '哈尔滨工业']
        filterlist = ['医', '护理']
        kl = 'wk'
        df = zy.somexx(xxsubstr=filterlist, kl=kl, cc='zk')
        pass

    return


def ys_zf(wh=100, zy=200):
    return wh*0.3 + zy*750/300*0.7


def ty_zf(wh=100, zy=100):
    return wh*0.3 + zy*750/100*0.7
