# -*- utf8 -*-


import pandas as pd
import pysal as ps
from dbfread import DBF


def dbf2df(dbfile, upper=True):
    "Read dbf file and return pandas DataFrame"
    with ps.open(dbfile) as db:  # I suspect just using open will work too
        df = pd.DataFrame({col: db.by_col(col) for col in db.header})
        if upper == True:
           df.columns = map(str.upper, db.header)
        return df


def read_dbf(dbfile):
    rownum  = 0
    for row in DBF(dbfile):
        if rownum < 10:
            print(row)
        rownum += 1
