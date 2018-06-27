

def lk(df, low, high):
    print(df[['xx', 'lkpos']][(df.lkpos <= high) & (df.lkpos >= low)])
    print(df[(df.lkpos <= high) & (df.lkpos >= low)].xx.apply(lambda  x: str(x).strip().ljust()))


def wk(df, low, high):
    print(df[['xx', 'wkpos']][(df.wkpos <= high) & (df.wkpos >= low)])
    print(df[(df.wkpos <= high) & (df.wkpos >= low)].xx)

