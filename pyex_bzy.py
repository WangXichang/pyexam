

def lk(df, low, high):
    print(df[['xx', 'lkpos']][(df.lkpos <= high) & (df.lkpos >= low)])


def wk(df, low, high):
    print(df[['xx', 'wkpos']][(df.wkpos <= high) & (df.wkpos >= low)])

