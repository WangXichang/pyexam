# _*_ utf-8 _*_


import time


def time_disper(fun):

    def dec_fun(*args, **kwargs):
        st = time.time()
        print('process start: {}'.format(fun))
        result = fun(*args, **kwargs)
        print('process[{}] elapsed time: {:.3f}'.format(fun, time.time() - st))
        return result

    return dec_fun
