# coding=utf8

import glob
import numpy as np
import weebar as wb
import matplotlib.pyplot as plt


def clip():
    file_list = glob.glob(pathname='f:/project/weebar/omr6/*.jpg')
    for fi, f in enumerate(file_list):
        bar = wb.testbar(filename=f, box_bottom=250, box_left=550, box_right=750, box_top=120)
        if bar.image_cliped is not None:
            plt.imsave('f:/project/weebar/bar128data/' + str(fi).zfill(3) + '.jpg',
                       bar.image_cliped)
            print('save {}'.format(f))


def wipe():
    file_list = glob.glob('f:/project/weebar/bar128data/*.jpg')
    for fi, f in enumerate(file_list):
        imc = plt.imread(f)
        im = rgb2gray(imc)
        imh = im.sum(axis=1)
        maxvalue = max(imh)
        imh = maxvalue - imh
        center = wb.BarcodeReader.proc1b_find_peak(imh)
        starline = center[0] - int(center[1]/2)
        im[starline: starline+10, :] = im[0:10,:]
        endline = center[0] + int(center[1]/2)
        im[endline-10:endline, :] = im[-11:-1, :]
        plt.imsave('f:/project/weebar/bar128_testdata/bar_test128_'+ str(fi).zfill(3) +'.jpg',
                   im)
        print(fi, f)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])