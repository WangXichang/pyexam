# coding: utf-8


import pandas as pd


class Bzy():
    def __init__(self):
        self.dflk = None
        self.dfwk = None

    def load(self):
        year = ['17', '18', '19']
        kl = ['like', 'wenke']
        path = 'f:/mywrite/newgk/3yeardata/'
        i = 0
        for y in year:
            dfl = pd.read_excel(path + 'zy20' + y + '-pt-bk1-like.xlsx', header=2)
            dfw = pd.read_excel(path + 'zy20' + y + '-pt-bk1-wenke.xlsx', header=2)
            dfl.loc[:, 'year'] = y
            dfw.loc[:, 'year'] = y
            if y == '17':
                self.dflk = dfl
                self.dfwk = dfw
            else:
                self.dflk = self.dflk.append(dfl)
                self.dfwk = self.dfwk.append(dfw)
        columns_mapper={'院校代号': 'yxdh',
                        '院校名称': 'yxmc',
                        '专业代号': 'zydh',
                        '专业名称': 'zymc',
                        '录取最低分': 'minscore',
                        '最低位次': 'minplace',
                        '平均分': 'meanscore'
                        }
        self.dfwk = self.dfwk.rename(columns=columns_mapper)
        self.dflk = self.dflk.rename(columns=columns_mapper)

        self.dfwk.minplace = self.dfwk.minplace.apply(lambda x: int(x[1:3]) if isinstance(x, str) else int(x))
        self.dflk.minplace = self.dflk.minplace.apply(lambda x: int(x[1:3]) if isinstance(x, str) else int(x))

    def find_wc(self, kl='like', xx='山东', zy=('经济'), wcrange=(100, 1000)):
        if kl == 'like':
            result = self.dflk.groupby(['yxdh', 'yxmc', 'zymc'])[['minplace', 'minscore']].mean()
        else:
            result = self.dfwk.groupby(['yxdh', 'yxmc', 'zymc'])[['minplace', 'minscore']].mean()
        print(result.head())
        result = result[[xx in x[1] for x in result.index]]
        for z in zy:
            result = result[[z in x[2] for x in result.index]]
        return result
