# -*- utf-8 -*-
# version 2018-06-19

import pandas as pd
import time


f = open('f:/students/juyunxia/mapfdata/mapfdata4.csv')

def read_large_csv(f):
    reader = pd.read_csv(f, sep=',', iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    start = time.time()
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    print('use time:{}'.format(time.time()-start))
    return df
