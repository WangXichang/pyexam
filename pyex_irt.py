# -*- utf-8 -*-
# version 2018
# update 2018-01-20


import scipy.stats as st
import matplotlib.pyplot as plt
# import seaborn as sn
# import numpy as np


def irt_response_curve():
    x = [v/100 for v in range(-400, 400)]
    y = [st.norm.cdf(v) for v in x]

    plt.plot(x, y)
    plt.xlabel(r'$\theta$')
    plt.ylabel('P')
    plt.show()
