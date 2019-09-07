# -*- utf8 -*-

import time
import sys
import os
import glob
import copy
from collections import Counter
from heapq import nlargest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
# from abc import ABCMeta, abstractclassmethod


def readbar(
        code_type='128c',
        file_list=(),
        box_top=None, box_left=None, box_bottom=None, box_right=None,
        display=False
        ):
    """
    :param code_type: 128a, 128b, 128c
    :param file_list: image files with barcode
    :param box_top: box top row in image
    :param box_left:   box left column in image
    :param box_bottom: box bottom row of image
    :param box_right:  box right column in image
    :param display: display messages in processing
    :return: result, object of BarcodeReader128
    """
    if not (isinstance(file_list, str) | isinstance(file_list, list)):
        print('file_list is not a valid type: list or str! type={}'.format(type(file_list)))
        return

    if isinstance(file_list, str):
        file_list = [file_list]

    st = time.time()
    reader = BarReaderFactory.create(code_type)  # BarcodeReader128()
    result_dataframe = None
    pg = ProgressBar(total=len(file_list), width=60)
    for fi, fs in enumerate(file_list):
        reader.get_dataframe(
            code_type=code_type,
            file_list=[fs],
            box_top=box_top, box_left=box_left, box_bottom=box_bottom, box_right=box_right,
            display=False)
        if fi == 0:
            result_dataframe = reader.result_dataframe[['file', 'code', 'valid', 'codelist']]
        else:
            reader.result_dataframe.index = [fi]
            result_dataframe = \
                result_dataframe.append(reader.result_dataframe[['file', 'code', 'valid', 'codelist']])
        if display:
            if fi == 0:
                print('index,   filename,    code,   codelist,    valid,   detect_steps')
            print(fi,
                  BarUtil.find_file_from_pathfile(file_list[fi]),
                  reader.result_code,
                  reader.result_codelist,
                  reader.result_code_valid,
                  reader.result_detect_steps
                  )
        pg.move()
        pg.log('')
    if display:
        print('total time:{:5.2n},  mean time:{:4.2n}'.
              format(time.time() - st, (time.time()-st) / len(file_list)))

    return result_dataframe


def testbar(
        code_type='128c',
        filename='',
        box_top=None, box_left=None, box_bottom=None, box_right=None,
        blur_kernel=(17, 17),
        display=False
        ):
    if not os.path.isfile(filename):
        print('file_list is not found!')
        return

    if not any([box_bottom, box_left, box_right, box_top]):
        im = plt.imread(filename)
        box_top = 0
        box_left = 0
        box_right = im.shape[1]     # column number
        box_bottom = im.shape[0]    # row number

    st = time.time()
    reader = BarReaderFactory.create(code_type)  # BarcodeReader128()
    reader.sys_test = True
    reader.new_image_blurr_template = blur_kernel
    reader.get_dataframe(
        code_type=code_type,
        file_list=[filename],
        box_top=box_top, box_left=box_left, box_bottom=box_bottom, box_right=box_right,
        display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time() - st, time.time()-st))

    if display:
        reader.show_bar_image()

    return reader


class BarReaderFactory(object):

    valid_code_type = ['128a', '128b', '128c', '39', '39asc']

    @staticmethod
    def create(code_type: str):
        if code_type.lower() in ['128', '128a', '128b', '128c']:
            return BarReader128()
        elif code_type.lower() in ['39', '39asc']:
            return BarReader39()
        else:
            print('not implemented code_type:{}'.format(code_type))
            print('valid code type={}'.format(BarReaderFactory.valid_code_type))
            raise Exception


class BarcodeReader(object):
    def __init__(self):
        # input data
        self.file_list = []
        self.code_type = ''

        # system para
        self.sys_test = False
        self.sys_step = 6

        # bar position in image
        self.box_top = 0
        self.box_bottom = 10000
        self.box_left = 0
        self.box_right = 10000

        # image processing parameters
        self.image_scan_scope = 12
        self.image_scan_step = 2
        self.image_scan_win_height = 5
        self.image_threshold_low = 10
        self.image_threshold_high = 110
        self.image_threshold_step = 6
        self.image_use_ratio = False
        self.image_ratio_row = 1
        self.image_ratio_col = 1
        self.new_image_blurr_template = (17, 15)

        # image data in procedure
        self.image_raw = None
        self.image_raw_cliped = None
        self.image_cliped = None
        self.image_gradient = None
        self.image_blurred = None
        self.image_closed = None
        self.image_bar = None
        self.image_bar01 = None
        self.image_bar_mid = 0
        self.image_bar_height = 0
        self.image_bar1 = None
        self.image_bar2 = None
        self.image_bar3 = None
        self.image_bar31 = None
        self.image_bar4 = None
        self.image_bar5 = None
        self.image_bar6 = None

        # bar data in procedure
        self.bar_pwlist_dict = {}
        self.bar_codecount_list = {}
        self.bar_collect_codecount_list = []
        self.bar_codelist_candidate_list = []
        self.bar_codelist_length = 0

        # result code and other data
        self.result_filename = ''
        self.result_code = ''
        self.result_codelist = []
        self.result_codelist_validity = []
        self.result_code_possible = []
        self.result_code_valid = False
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = 0
        self.result_dataframe = None

        # tools
        self.checker = None
        self.decoder = None
        self.adjuster = None
        self.filler = None
        self.pruner = None
        self.compounder = None

    def set_image_files(self, file_list):
        self.file_list = file_list

    def set_image_box(self,
                      box_top=None, box_bottom=None, box_left=None, box_right=None
                      ):
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.box_left = box_left
        self.box_right = box_right

    def set_image_ratio(self, ratio_row=None, ratio_col=None):
        if ratio_row is not None:
            self.image_ratio_row = ratio_row
        if ratio_col is not None:
            self.image_ratio_col = ratio_col

    def show_raw_image(self):
        plt.figure('raw image')
        plt.imshow(self.image_raw)

    def show_bar_image(self):
        image_bar = [self.image_bar1, self.image_bar2,
                     self.image_bar3, self.image_bar31, self.image_bar4,
                     self.image_bar5, self.image_bar6, self.image_cliped,
                     self.image_raw_cliped]
        image_bar_index = [1, 2, 3, 31, 4, 5, 6, 7, 9]
        image_bar_method = ['bar image-1', 'amplify 1.15x1.2', 'amplify 1.2x1.5',
                            'amplify {}x{}'.format(self.image_ratio_row, self.image_ratio_col),
                            'extend scope',
                            'blur kernel({},{})'.format(*self.new_image_blurr_template),
                            'reverse image', 'cliped image', 'raw cliped'
                            ]
        null_image = np.zeros((5, 5)) * 200
        plt.figure('gray bar image:'+self.result_filename+ '  '+self.result_code)
        # plt.subplot(23)
        for i in range(9):
            ploted = 0
            plt.subplot(3, 3, i + 1)
            if image_bar[i] is not None:
                if image_bar[i].shape[0]*image_bar[i].shape[1] > 0:
                    plt.imshow(image_bar[i])
                    ploted = 1
            if ploted == 0:
                plt.imshow(null_image)
            plt.xlabel('{}'.format(image_bar_method[i]))
            plt.title('image bar {}'.format(image_bar_index[i]))
        # plt.imshow(self.image_bar)

    def show_clip_image(self):
        plt.imshow(self.image_cliped)
        plt.show()

    def get_dataframe(
            self,
            code_type=None,
            file_list=None,
            box_top=None, box_left=None, box_bottom=None, box_right=None,
            display=False
            ):

        # set image files to read
        if type(file_list) == list:
            self.file_list = file_list
        elif type(file_list) == str:
            self.file_list = [file_list]
        else:
            print('invalid files {}'.format(file_list))
        # set code_type
        if isinstance(code_type, str):
            self.code_type = code_type

        # read barcode to dataframe from imagefiles(file_list)
        for i, _file in enumerate(file_list):
            self.get_result(
                image_file=_file,
                code_type=code_type,
                box_top=box_top, box_bottom=box_bottom, box_left=box_left, box_right=box_right,
                display=display
                )
            if i > 0:
                self.result_dataframe = \
                    self.result_dataframe.append(
                        pd.DataFrame({
                            'file': [BarUtil.find_file_from_pathfile(_file)],
                            'code': [self.result_code],
                            'codelist': [self.result_codelist],
                            'validity': [self.result_codelist_validity],
                            'valid': [self.result_code_valid],
                            'steps': [self.result_detect_steps],
                            'found': [self.result_barimage_found],
                            'fill': [self.result_fill_loss]
                            }, index=[i]))
            else:
                self.result_dataframe = \
                    pd.DataFrame({
                        'file': [BarUtil.find_file_from_pathfile(_file)],
                        'code': [self.result_code],
                        'codelist': [self.result_codelist],
                        'validity': [self.result_codelist_validity],
                        'valid': [self.result_code_valid],
                        'steps': [self.result_detect_steps],
                        'found': [self.result_barimage_found],
                        'fill': [self.result_fill_loss]
                        }, index=[i])
            if display:
                print(i,
                      BarUtil.find_file_from_pathfile(_file),
                      self.result_code,
                      self.result_codelist,
                      self.result_code_valid
                      )

    def get_result(self,
                   code_type=None,
                   image_file='',
                   box_top=None, box_bottom=None, box_left=None, box_right=None,
                   display=False
                   ):

        # initiate result data
        self.result_filename = image_file
        self.result_code_valid = False
        self.result_code = ''
        self.result_codelist = []
        self.result_code_possible = []
        self.result_codelist_validity = []
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = False

        # read image to self.image_raw from image_file
        if not self.proc0_read_image_from_file(image_file=image_file, display=display):
            if display:
                print('not found image file:{}'.format(image_file))
            return

        self.main_proc(
            image_data=self.image_raw,
            code_type=code_type,
            box_top=box_top, box_bottom=box_bottom, box_left=box_left, box_right=box_right,
            ratio_row=None, ratio_col=None,
            image_threshold_low=None, image_threshold_high=None, image_threshold_step=None,
            image_scan_scope=None, image_scan_step=None, image_scan_line_num=None,
            display=display
            )
        return
        # end get_result

    def main_proc(
            self,
            code_type=None,
            image_data=None,
            box_top=None, box_left=None, box_bottom=None, box_right=None,
            ratio_row=None, ratio_col=None,
            image_threshold_low=None, image_threshold_high=None, image_threshold_step=None,
            image_scan_scope=None, image_scan_step=None, image_scan_line_num=None,
            display=False
            ):

        # set bar area
        if box_top is not None:
            self.box_top = box_top
        if box_left is not None:
            self.box_left = box_left
        if box_bottom is not None:
            self.box_bottom = box_bottom
        if box_right is not None:
            self.box_right = box_right

        # set other para
        if type(image_threshold_low) == int:
            self.image_threshold_low = image_threshold_low
        if type(image_threshold_high) == int:
            self.image_threshold_high = image_threshold_high
        if type(image_threshold_step) == int:
            self.image_threshold_step = image_threshold_step
        if type(image_scan_scope) == int:
            self.image_scan_scope = image_scan_scope
        if type(image_scan_step) == int:
            self.image_scan_step = image_scan_step
        if type(image_scan_line_num) == int:
            self.image_scan_win_height = image_scan_line_num

        # initiate proc var
        self.bar_pwlist_dict = {}
        self.bar_codecount_list = {}
        self.bar_collect_codecount_list = []
        self.bar_codelist_length = 0
        self.bar_codelist_candidate_list = []

        # initiate result var
        self.result_codelist = []
        self.result_code = ''
        self.result_code_valid = False
        self.result_code_possible = []
        self.result_codelist_validity = []
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = False
        get_result = False

        # get bar image
        self.image_raw_cliped = image_data[box_top: box_bottom+1, box_left: box_right+1]
        if not self.proc1_get_barimage(image_data=image_data, display=display):
            if display:
                print('fail to extract bar from raw image!')
            return False

        # check empty bar_image
        if len(self.image_bar.shape) == 1:
            if display:
                print('--- 0th detect as bar image empty ---')
            return False
        if self.image_bar.shape[0] * self.image_bar.shape[1] == 0:
            if display:
                print('--- 0th detect as bar image empty ---')
            return False

        # first check barcode
        self.result_detect_steps += 1
        if display:
            print('--- 1th detect with raw bar image ---')
        self.proc2_get_codelist(code_type=code_type, display=display)
        if self.proc3_get_resultcode():   # display=display):
            get_result = True
        self.image_bar1 = copy.copy(self.image_bar)

        # second check barcode by amplify image
        if (not get_result) | ('**' in self.result_codelist):  # ('*' in ''.join(self.result_codelist)):
            self.result_detect_steps += 1
            if display:
                print('--- 2th detect with amplified bar image({0}, {1}) ---'.format(1.15, 1.25))
            self.image_bar = BarUtil.image_amplify(self.image_bar, ratio_row=1.15, ratio_col=1.25)
            self.proc2_get_codelist(code_type=code_type, display=display)
            if self.proc3_get_resultcode():    # display=display):
                get_result = True
            if self.sys_test:
                self.image_bar2 = copy.copy(self.image_bar)

        # the third check by amplify image
        if (not get_result) | ('**' in self.result_codelist):  # ('*' in ''.join(self.result_codelist)):
            self.result_detect_steps += 1
            if display:
                print('--- 3th detect with amplified bar image({0}, {1}) ---'.format(1.2, 1.5))
            self.image_bar = BarUtil.image_amplify(self.image_bar1, ratio_row=1.2, ratio_col=1.5)
            self.proc2_get_codelist(code_type=code_type, display=display)
            if self.proc3_get_resultcode():    # display=display):
                get_result = True
            if self.sys_test:
                self.image_bar3 = copy.copy(self.image_bar)

        # detect 3+1  by amplify image
        if (not get_result) & ((ratio_row is not None) | (ratio_col is not None)):
            ratio_row = 1 if ratio_row is None else ratio_row
            ratio_col = 1 if ratio_col is None else ratio_col
            self.result_detect_steps += 1
            if display:
                print('--- 3th+ detect with amplified bar image({0}, {1}) ---'.format(ratio_row, ratio_col))
            self.image_bar = BarUtil.image_amplify(self.image_bar, ratio_row=ratio_row, ratio_col=ratio_col)
            self.proc2_get_codelist(code_type=code_type, display=display)
            if self.proc3_get_resultcode():     # display=display):
                get_result = True
            if self.sys_test:
                self.image_bar31 = copy.copy(self.image_bar)

        # the fourth check by extend scan_scope
        old_scope = self.image_scan_scope
        new_scope = int(self.image_bar.shape[0]/2) - 15
        if (not get_result) & (new_scope > 15):
            if display:
                print('--- 4th check with extend scan scope to {} ---'.format(new_scope))
            self.bar_collect_codecount_list = []
            self.image_scan_scope = new_scope
            self.proc2_get_codelist(code_type=code_type, display=display)
            self.result_detect_steps += 1
            self.image_scan_scope = old_scope
            if self.proc3_get_resultcode():
                get_result = True

        # the fourth check by extend scan_scope
        # self.new_image_blurr_template = (17, 15)
        if (not get_result) & \
                (self.image_cliped.shape[0] > 50) & (self.image_cliped.shape[1] > 80):
            if display:
                print('--- 5th check with new blurr template to {} ---'.format(self.new_image_blurr_template))
            self.result_detect_steps += 1
            self.bar_collect_codecount_list = []
            self.proc1_get_barimage(image_data=self.image_raw,
                                    image_blur_kernel=self.new_image_blurr_template)
            self.proc2_get_codelist(code_type=code_type, display=display)
            if self.proc3_get_resultcode():
                get_result = True
            if self.sys_test:
                self.image_bar5 = copy.copy(self.image_bar)

        # reverse bar image
        if (not get_result) & \
           (self.image_cliped.shape[0] > 50) & (self.image_cliped.shape[1] > 80):
            if display:
                print('--- 6th check with reversing ---')
            # save_image = copy.copy(self.image_bar)
            self.result_detect_steps += 1
            self.bar_collect_codecount_list = []
            # self.proc1_get_barimage(image_data=self.image_raw,
            #                        image_blur_kernel=new_image_blurr_template)
            self.image_bar = self.matrix_col_reverse(self.image_bar1)
            self.proc2_get_codelist(code_type=code_type, display=display)
            if self.proc3_get_resultcode():
                get_result = True
            if self.sys_test:
                self.image_bar6 = copy.copy(self.image_bar)

        if not get_result:
            self.result_code = ''

        if display:
            print('--' * 60,
                  '\n result_code = {0}'
                  '\n    codelist = {1}'
                  '\npossiblecode = {2}'
                  '\n    validity = {3}'
                  '\nbarimagfound = {4}'
                  '\n detectsteps = {5}'.
                  format(self.result_code,
                         self.result_codelist,
                         self.result_code_possible,
                         self.result_codelist_validity,
                         self.result_barimage_found,
                         self.result_detect_steps))

        return True
        # end get_barcode

    @staticmethod
    def matrix_row_reverse(matrix_2d):
        return matrix_2d[::-1]

    @staticmethod
    def matrix_col_reverse(matrix_2d):
        return np.array([matrix_2d[r, :][::-1] for r in range(matrix_2d.shape[0])])

    # get image_raw from filename
    def proc0_read_image_from_file(self, image_file, display=False):
        # read image data from image file
        if (type(image_file) != str) or (image_file == ''):
            if display:
                print('no image file given!')
            return False
        else:
            if not os.path.isfile(image_file):
                print('not found file: %s' % image_file)
                return False
        # read image,  from self.image_filenames
        self.image_raw = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
        return True

    # get image_bar from self.image_raw
    def proc1_get_barimage(self,
                           image_data=None,
                           image_blur_kernel=(11, 11),
                           display=False):
        # check self.image_raw
        if image_data is None:
            if display:
                print('image_data is None!')
            return False

        # clip image by box_
        result, self.image_cliped, self.box_left, self.box_right, self.box_top, self.box_bottom = \
            self.proc1a_adjust_clip(image_data=image_data,
                                    box_left=self.box_left,
                                    box_right=self.box_right,
                                    box_top=self.box_top,
                                    box_bottom=self.box_bottom,
                                    display=display)
        if not result:
            return False
        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        gradx = cv2.Sobel(self.image_cliped, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grady = cv2.Sobel(self.image_cliped, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradx, grady)
        self.image_gradient = cv2.convertScaleAbs(gradient)
        # blur by Gauss kenerl, convolve by 2D Gauss function
        self.image_blurred = cv2.blur(gradient, image_blur_kernel)     # default (11, 11)
        (_, thresh) = cv2.threshold(self.image_blurred, 225, 255, cv2.THRESH_BINARY)
        # link neighbor block and remove noise by morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        self.image_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.image_closed = cv2.erode(self.image_closed, None, iterations=3)
        self.image_closed = cv2.dilate(self.image_closed, None, iterations=5)
        # get 8UC1 format image, uint format
        img = cv2.normalize(self.image_closed,
                            None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8UC1)
        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        (_, contours, __) = cv2.findContours(
                                img.copy(),
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv2.boxPoints(rect))
            # print(box)
            left, top, right, bottom = \
                min(box[:, 0]) - 15 if min(box[:, 0]) > 15 else min(box[:, 0]), \
                min(box[:, 1]) - 15 if min(box[:, 1]) > 15 else min(box[:, 1]), \
                max(box[:, 0]) + 15, \
                max(box[:, 1]) + 15
            # get bar image from box area
            self.image_bar = self.image_cliped[top:bottom, left:right]
            self.result_barimage_found = True
        # image_bar is empty instead of using image_cliped
        else:
            self.image_bar = self.image_cliped.copy()
            self.result_barimage_found = False

        # get mid row loc
        if self.result_barimage_found:
            cl = (255-self.image_bar).sum(axis=1)
            self.image_bar_mid, self.image_bar_height = \
                self.proc1b_find_peak(cl)
        else:
            # no bar image or peak not found
            self.image_bar_mid = int(self.image_cliped.shape[0] / 2)

        return True

    def proc1a_adjust_clip(self,
                           image_data,
                           box_left=None,
                           box_right=None,
                           box_top=None,
                           box_bottom=None,
                           display=False):
        if all([box_right is None, box_left is None, box_top is None, box_bottom is None]):
            return False, image_data, box_left, box_right, box_top, box_bottom
        move_step = 10
        gap = 20
        # define conditions to move
        move_to_right = '(bar_center[0] + bar_wid[0]/2 + gap) > image_cliped.shape[1]'
        move_to_left = '(bar_center[0] - bar_wid[0] / 2 - gap) < 0'
        move_to_up = '(bar_center[1] - bar_wid[1] / 2 - gap) < 0'
        move_to_down = '(bar_center[1] + bar_wid[1] / 2 + gap) > image_cliped.shape[0]'
        # first loc bar
        image_cliped = image_data[box_top: box_bottom+1, box_left: box_right+1]
        bar_center, bar_wid = self.proc1a1_get_barimage_center_and_width(255 - image_cliped)
        step = 0
        move_condition = eval(move_to_right) | eval(move_to_left) | eval(move_to_up) | eval(move_to_down)
        while move_condition & (step < 21):
            move_right = move_step if eval(move_to_right) else 0
            move_left = move_step if eval(move_to_left) else 0
            move_up = move_step if eval(move_to_up) else 0
            move_down = move_step if eval(move_to_down) else 0
            image_cliped, box_left, box_right, box_top, box_bottom = \
                self.proc1a2_box_move(image_data=image_data,
                                      move_right=move_right, move_left=move_left,
                                      move_up=move_up, move_down=move_down,
                                      box_left=box_left, box_right=box_right,
                                      box_top=box_top, box_bottom=box_bottom)
            bar_center, bar_wid = self.proc1a1_get_barimage_center_and_width(255 - image_cliped)
            step += 1
            if display:
                dstr = ('' if move_left == 0 else '  move_left={}, left={}'.format(move_left, box_left)) + \
                       ('' if move_right == 0 else '  move_right={}, right={}'.format(move_right, box_right)) + \
                       ('' if move_up == 0 else '  move_up={}'.format(move_up)) +\
                       ('' if move_down == 0 else '  move_down={}'.format(move_down))
                print('extend bar image at step={0}, {1} '.format(step, dstr))
            # print(bar_center, bar_wid, self.image_cliped.shape)
            move_condition = (eval(move_to_right) | eval(move_to_left) | eval(move_to_up) | eval(move_to_down))
        # print(image_cliped.shape)
        top_gap = bar_center[1] - int(bar_wid[1]/2)
        bottom_gap = image_cliped.shape[0] - int(bar_center[1] + bar_wid[1]/2)
        left_gap = bar_center[0] - int(bar_wid[0]/2)
        right_gap = image_cliped.shape[1] - int(bar_center[0]+bar_wid[0]/2)
        clip_top = top_gap-gap-10 if top_gap > 45 else 0
        clip_bottom = top_gap-gap-10 if bottom_gap > 45 else 0
        clip_left = top_gap-gap-10 if left_gap > 45 else 0
        clip_right = top_gap-gap-10 if right_gap > 45 else 0
        image_cliped = image_data[box_top+clip_top: box_bottom+1-clip_bottom,
                                  box_left+clip_left: box_right + 1-clip_right]
        # print('tblr:', top_gap, bottom_gap, left_gap, right_gap)
        return True, image_cliped, box_left, box_right, box_top, box_bottom

    def proc1a1_get_barimage_center_and_width(self, bar_image):
        horizon_fun = bar_image.sum(axis=0)
        horizon_mid, horizon_wid = self.proc1b_find_peak(horizon_fun)
        vertical_fun = bar_image.sum(axis=1)
        vertical_mid, vertical_wid = self.proc1b_find_peak(vertical_fun)
        if (horizon_mid > 0) & (vertical_mid > 0):
            return (horizon_mid, vertical_mid), (horizon_wid, vertical_wid)
        return (-1, -1), (0, 0)

    @staticmethod
    def proc1a2_box_move(image_data,
                         move_right=0, move_up=0, move_left=0, move_down=0,
                         box_left=0, box_right=0, box_top=0, box_bottom=0):
        _box_left = box_left - move_left
        if box_left < 0:
            _box_left = 0
        _box_right = box_right + move_right
        _box_top = box_top - move_up
        if box_top < 0:
            _box_top = 0
        _box_bottom = box_bottom + move_down
        image_cliped = image_data[_box_top: _box_bottom,
                                  _box_left: _box_right]
        return image_cliped, _box_left, _box_right, _box_top, _box_bottom

    @staticmethod
    def proc1b_find_peak(mapfun):
        # mean_lift_ratio = 1.62
        if len(mapfun) == 0:
            return 0, 0
        map_min = np.min(mapfun)
        mapfun = mapfun - map_min
        map_mean = np.mean(mapfun)
        _peak = np.where(mapfun > map_mean)[0]
        # _peak = np.where(mapfun > map_mean * mean_lift_ratio)[0]
        mid, wid = -1, 0
        if len(_peak) > 0:
            # found peak from cl_peak[0] to cl_peak[-1]
            mid = int((_peak[0] + _peak[-1]) / 2)
            wid = _peak[-1] - _peak[0]
        return mid, wid

    # get cnadidate_codelist_list: get_pwlist, get_codecount, get_collect_codecount
    def proc2_get_codelist(self, code_type, display=False):
        # scan for gray_mean+ gray_threshold_low to high
        # barimage-->pwlist-->codecount-->collect_codecount
        for th_gray in range(self.image_threshold_low,
                             self.image_threshold_high,
                             self.image_threshold_step):
            # step-1: get pwlist
            self._proc21_get_pwlist_from_barimage(gray_shift=th_gray)
            # step-2: get codecount from pwlist
            if self._proc22_get_codecount_from_pwlist(code_type=code_type, th_gray=th_gray, display=display):
                # step-3: collect codecount from bar_codecount
                self._proc23_get_collect_codecount(codecount_list=self.bar_codecount_list)
        if display:
            print('collect codecount:{}'.format(self.bar_collect_codecount_list))
        # step4: pruning start,stop,empty_tail
        self.bar_collect_codecount_list = self.pruner(self.bar_collect_codecount_list)
        if display:
            print('pruned collect:{}'.format(self.bar_collect_codecount_list))
        # step5: get candidate_codelist from collect_codecount_list
        self.bar_codelist_candidate_list = self._proc24_get_candidate_codelist()
        return

    def _proc21_get_pwlist_from_barimage(self, gray_shift=60):
        # using para: self.image_detect_win_high, scan_scope, gray_shift
        # get binary image
        # print(self.image_bar.shape)
        if len(self.image_bar.shape) == 1:
            return
        if self.image_bar.shape[0] * self.image_bar.shape[1] == 0:
            return

        # reverse image: black to white
        img = 255 - self.image_bar.copy()

        # get binary image: bar01
        th = img.mean() + gray_shift
        img[img < th] = 0
        img[img > 0] = 1    # pixels for img>=th
        self.image_bar01 = img

        # get bar bar&space width list
        # mid_loc = self.image_bar_mid
        # scan from mid - scope to mid + scope by step=scan_step
        for _line in range(-self.image_scan_scope,
                           self.image_scan_scope,
                           self.image_scan_step):
            row = self.image_bar_mid + _line
            img_line = np.around(self.image_bar01[row: row + self.image_scan_win_height, :].sum(axis=0) /
                                 self.image_scan_win_height, decimals=0)
            # trip head & tail 0
            for j in range(len(img_line)):
                if img_line[j] == 1:
                    img_line = img_line[j:]
                    break
            for j in range(len(img_line) - 1, 0, -1):
                if img_line[j] == 1:
                    img_line = img_line[:j + 1]
                    break
            # get pwlist
            bs_wid_list, lastc, curwid = [], img_line[0], 1
            for cc in img_line[1:]:
                if cc != lastc:
                    bs_wid_list.append(curwid)
                curwid = curwid + 1 if cc == lastc else 1
                lastc = cc
            bs_wid_list.append(curwid)
            self.bar_pwlist_dict[(gray_shift, _line)] = bs_wid_list
        return  # _get2x1

    # get codeCountDict from pwlist for a fixed th_gray and all scanning line
    # from bar_bspixel_list_dict[th_gray, -scope:scope]
    def _proc22_get_codecount_from_pwlist(self, code_type=None, th_gray=0, display=False):
        mlines_codelist_dict = dict()
        for line_no in range(-self.image_scan_scope,
                             self.image_scan_scope,
                             self.image_scan_step):
            # decode from pwlist
            result = self.decoder(self.bar_pwlist_dict[th_gray, line_no], code_type)
            if len(result) > 0:
                mlines_codelist_dict[line_no] = result

        # record to codecount
        if len(mlines_codelist_dict) > 0:
            max_len = max([len(mlines_codelist_dict[x]) for x in mlines_codelist_dict])
        else:
            return False
        self.bar_codecount_list = [{} for _ in range(max_len)]
        for line_no in range(-self.image_scan_scope,
                             self.image_scan_scope,
                             self.image_scan_step):
            # not valid line or invalid bs list(no data items)
            if (line_no not in mlines_codelist_dict) or (len(mlines_codelist_dict[line_no]) < 4):
                continue
            for di, dc in enumerate(mlines_codelist_dict[line_no]):
                # if dc.isdigit() or (dc in ['StartC', 'Stop', 'CodeB']):
                if '**' not in dc:
                    if dc in self.bar_codecount_list[di]:
                        self.bar_codecount_list[di][dc] += 1
                    else:
                        self.bar_codecount_list[di][dc] = 1

        if display:
            print('scan th_gray={}:'.format(th_gray), self.bar_codecount_list)

        # abandon codelist with 4 or more empty code items
        # if sum([0 if len(d) > 0 else 1 for d in self.bar_codecount_list]) > 3:
        #    return False

        return True

    # get bar_collect_codecount_list from bar_codecount_list
    def _proc23_get_collect_codecount(self, codecount_list):
        if len(self.bar_collect_codecount_list) == 0:
            # first time to initiate
            self.bar_collect_codecount_list = codecount_list
            self.bar_codelist_length = len(codecount_list)
        else:
            # others add
            self.bar_codelist_length = (self.bar_codelist_length + len(codecount_list))/2
            for i, dc in enumerate(codecount_list):
                if i < len(self.bar_collect_codecount_list):
                    for kc in dc:
                        if kc in self.bar_collect_codecount_list[i]:
                            self.bar_collect_codecount_list[i][kc] += dc[kc]
                        else:
                            self.bar_collect_codecount_list[i][kc] = dc[kc]
                else:
                    self.bar_collect_codecount_list.append(dc)
        return

    # get candidate_codelist from collent_codecount_list
    def _proc24_get_candidate_codelist(self):
        # empty or invalid collect
        if len(self.bar_collect_codecount_list) < 3:
            return []
        # empty codecount number > 3
        if sum([1 for d in self.bar_collect_codecount_list if len(d) == 0]) > 3:
            return []

        # select maxcount codelist
        # maybe more than one if too many maxvalue items exist
        codecount_list = copy.deepcopy(self.bar_collect_codecount_list)
        result_lists = []
        check_valid = False
        while True:
            loop_break = True
            select_codelist = []
            for ci, cd in enumerate(codecount_list):
                if len(cd) == 0:
                    select_codelist.append('**')
                    continue
                mv = max(list(cd.values()))
                md = [d for d in cd if cd[d] == mv]
                select_codelist.append(md[0])
                if len(md) > 1:
                    codecount_list[ci].pop(md[0])
                    loop_break = False
            result_lists.append(select_codelist)
            if self.checker(codelist=select_codelist, code_type=self.code_type)[0]:
                check_valid = True
                break
            if loop_break:
                break

        # add  new codelist by selecting other checkcode
        ckdict = codecount_list[-2]
        if (not check_valid) & (len(ckdict) > 0):
            result_lists2 = result_lists.copy()  # avoid endless loop
            for code_list in result_lists:
                if '**' not in code_list:
                    for d in ckdict:
                        newlist = code_list[0:-2] + [d] + [code_list[-1]]
                        if self.checker(codelist=newlist, code_type=self.code_type)[0]:
                            result_lists2.append(newlist)
            result_lists = result_lists2

        result_lists = self.adjuster(result_lists, self.code_type)

        return result_lists

    # get result code from self.bar_codelist_candidate_list
    def proc3_get_resultcode(self):
        """
        # select code if no '**' in code, checkcode
        self.result_code_valid = False
        for code_list in self.bar_codelist_candidate_list:
            code_len = len(code_list)
            # invalid codelist length
            if code_len < 3:
                continue
            # verify checkcode and return result
            check_code = code_list[code_len - 2]
            # print(code_list)
            if (check_code.isdigit()) & ('**' not in code_list[1:code_len-1]):
                # ('*' not in ''.join(code_list[1:code_len-1])):
                if display:
                    print('no loss candidate:', code_list)
                if self.checker(code_list, self.code_type)[0]:
                    self.make_code_from_codelist(codelist=code_list)
                    return True
        """

        # computing max score codelist from candidate if no valid codelist
        result_codelist0 = self.get3x1_maxscore_codelist_from_candidate(self.bar_codelist_candidate_list)
        if len(result_codelist0) > 3:
            # self.make_code_from_codelist(codelist=result_codelist0)
            self.result_code_valid = self.checker(codelist=result_codelist0,
                                                  code_type=self.code_type)[0]
            self.result_code = self.compounder(codelist=result_codelist0,
                                               code_type=self.code_type)
            self.result_codelist = result_codelist0
            self.result_codelist_validity = self.make_codelist_validity(result_codelist0)

        return self.result_code_valid

    def make_code_from_codelist(self, codelist):
        self.result_code_valid = self.checker(codelist=codelist,
                                              code_type=self.code_type)[0]
        self.result_code = self.compounder(codelist, self.code_type)
        self.result_codelist = codelist
        self.result_codelist_validity = self.make_codelist_validity(codelist)
        return

    def make_codelist_validity(self, codelist):
        cdvalidity = [-1 for _ in codelist]
        colen = len(self.bar_collect_codecount_list)
        for ci, cd in enumerate(codelist):
            if ci in range(colen):
                if cd in self.bar_collect_codecount_list[ci]:
                    cdvalidity[ci] = self.bar_collect_codecount_list[ci][cd]
                    continue
        sm = np.mean(cdvalidity)
        if sm > 100:
            return [eval('{:.4}'.format(v / sm)) for v in cdvalidity]
        else:
            return [eval('{:.4}'.format(v/100)) for v in cdvalidity]

    # think to deprecate
    def get4_result_by_fill_lowvalidity(self, display=False):
        if display:
            print('check validity:{}'.format(self.result_codelist_validity))
        if len(self.result_codelist_validity) == 0:
            return False
        lowvalidity = min(self.result_codelist_validity)
        if lowvalidity > 0.3:
            if display:
                print('min validity({})>0.3, abandon to fill'.format(lowvalidity))
            return False
        maxvalidity = max(self.result_codelist_validity)
        if maxvalidity < 0.1:
            if display:
                print('validity({}) is too little, abandon to fill'.format(maxvalidity))
            return False
        loc = self.result_codelist_validity.index(lowvalidity)
        if 0 < loc < len(self.result_codelist) - 2:  # think, not include checkcode
            newcodelist = self.result_codelist.copy()
            newcodelist[loc] = '**'
            # filled_codelists = self.bar_128_fill_loss(newcodelist[1:-2])
            filled_codelists = self.filler(newcodelist, self.code_type)
            if len(filled_codelists) > 0:
                if len(filled_codelists) > 1:
                    self.result_code_possible = filled_codelists[1:]
                codelist = filled_codelists[0]
                self.make_code_from_codelist(codelist=codelist)
                self.result_fill_loss += 1
                if display:
                    print('finish filling:{0} '
                          '\n          from:{1}'
                          '\n   result code:{2}'.
                          format(self.result_codelist, newcodelist, self.result_code))
                return True
        return False

    # select codelist with big score(occurrence times)
    def get3x1_maxscore_codelist_from_candidate(self, codelist_list):
        if len(codelist_list) == 0:
            return []
        result_codelist_score = [0 for _ in codelist_list]
        # maxscore = 0
        for ci, cl in enumerate(codelist_list):
            if len(cl) != len(self.bar_collect_codecount_list):
                continue
            score = 0
            for di, dk in enumerate(cl):
                if dk in self.bar_collect_codecount_list[di]:
                    score += self.bar_collect_codecount_list[di][dk]
            result_codelist_score[ci] = score
            # maxscore = max(maxscore, score)
        maxscore = max(result_codelist_score)
        return codelist_list[result_codelist_score.index(maxscore)]


class ProgressBar:
    """
    Usage: 1. get object: pb = ProgressBar(totlal=length, display_gap=10)
           2. call pb in loop:
            pg.move()
            pg.log(dispmessage)
           3. call in end outside loop:
            pg.count=length
            pg.log(diapmessage)
    """
    def __init__(self, count=0, total=0, width=100, display_gap=1):
        self.count = count
        self.total = total
        self.width = width
        self.display_gap = display_gap

    def move(self):
        self.count += 1

    def log(self, s=''):
        if (self.count != 1) & (self.count != self.total) & \
                (self.count % self.display_gap > 0):
            return
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        if len(s) > 0:
            print(s)
        progress = int(self.width * self.count / self.total)
        progress = progress if progress < self.width else progress + 1
        # black:\u2588
        black_part_str = '\u25A3' * progress
        white_part_str = '\u2610' * (self.width - progress)
        sys.stdout.write('{0:6d}/{1:d} {2:>6}%: '.
                         format(self.count, self.total, str(round(self.count/self.total*100, 2))))
        sys.stdout.write(black_part_str + ('\u27BD' if progress < self.width else '') +
                         white_part_str + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


class BarReader128(BarcodeReader):

    def __init__(self):
        super().__init__()
        # set new decoder for 128
        self.decode_worker = BarDecoderFactory.create('128')
        self.checker = self.decode_worker.check
        self.decoder = self.decode_worker.decode
        self.pruner = self.decode_worker.prune
        self.adjuster = self.decode_worker.adjust
        self.filler = self.decode_worker.fill
        self.compounder = self.decode_worker.compound


class BarReader39(BarcodeReader):

    def __init__(self):
        super().__init__()
        # set new decoder for 128
        self.decode_worker = BarDecoderFactory.create('39')
        self.checker = self.decode_worker.check
        self.decoder = self.decode_worker.decode
        self.pruner = self.decode_worker.prune
        self.adjuster = self.decode_worker.adjust
        self.filler = self.decode_worker.fill
        self.compounder = self.decode_worker.compound


class BarConstant:
    table_code39 = \
        '''A,110101001011
        B,101101001011
        C,110110100101
        D,101011001011
        E,110101100101
        F,101101100101
        G,101010011011
        H,110101001101
        I,101101001101
        J,101011001101
        K,110101010011
        L,101101010011
        M,110110101001
        N,101011010011
        O,110101101001
        P,101101101001
        Q,101010110011
        R,110101011001
        S,101101011001
        T,101011011001
        U,110010101011
        V,100110101011
        W,110011010101
        X,100101101011
        Y,110010110101
        Z,100110110101
        0,101001101101
        +,100101001001
        1,110100101011
        -,100101011011
        2,101100101011
        *,100101101101
        3,110110010101
        /,100100101001
        4,101001101011
        %,101001001001
        5,110100110101
        $,100100100101
        6,101100110101
        .,110010101101
        7,101001011011
         ,100110101101
        8,110100101101
        9,101100101101'''

    table_code128 = \
        '''2;1;2;2;2;2;//sp// =  //00
        2;2;2;1;2;2;// !// =   //01
        2;2;2;2;2;1;// "// =   //02
        1;2;1;2;2;3;// #// =   //03
        1;2;1;3;2;2;// $// =   //04
        1;3;1;2;2;2;// %// =   //05
        1;2;2;2;1;3;// &// =   //06
        1;2;2;3;1;2;//...// =  //07
        1;3;2;2;1;2;// (// =   //08
        2;2;1;2;1;3;// )// =   //09
        2;2;1;3;1;2;// *// =   //10
        2;3;1;2;1;2;// +// =   //11
        1;1;2;2;3;2;// ,// =   //12
        1;2;2;1;3;2;// -// =   //13
        1;2;2;2;3;1;// .// =   //14
        1;1;3;2;2;2;// / // =   //15
        1;2;3;1;2;2;// 0// =   //16
        1;2;3;2;2;1;// 1// =   //17
        2;2;3;2;1;1;// 2// =   //18
        2;2;1;1;3;2;// 3// =   //19
        2;2;1;2;3;1;// 4// =   //20
        2;1;3;2;1;2;// 5// =   //21
        2;2;3;1;1;2;// 6// =   //22
        3;1;2;1;3;1;// 7// =   //23
        3;1;1;2;2;2;// 8// =   //24
        3;2;1;1;2;2;// 9// =   //25
        3;2;1;2;2;1;// :// =   //26
        3;1;2;2;1;2;// ;// =   //27
        3;2;2;1;1;2;// <// =   //28
        3;2;2;2;1;1;// =// =   //29
        2;1;2;1;2;3;// >// =   //30
        2;1;2;3;2;1;// ?// =   //31
        2;3;2;1;2;1;// @// =   //32
        1;1;1;3;2;3;// A// =   //33
        1;3;1;1;2;3;// B// =   //34
        1;3;1;3;2;1;// C// =   //35
        1;1;2;3;1;3;// D// =   //36
        1;3;2;1;1;3;// E// =   //37
        1;3;2;3;1;1;// F// =   //38
        2;1;1;3;1;3;// G// =   //39
        2;3;1;1;1;3;// H// =   //40
        2;3;1;3;1;1;// I// =   //41
        1;1;2;1;3;3;// J// =   //42
        1;1;2;3;3;1;// K// =   //43
        1;3;2;1;3;1;// L// =   //44
        1;1;3;1;2;3;// M// =   //45
        1;1;3;3;2;1;// N// =   //46
        1;3;3;1;2;1;// O// =   //47
        3;1;3;1;2;1;// P// =   //48
        2;1;1;3;3;1;// Q// =   //49
        2;3;1;1;3;1;// R// =   //50
        2;1;3;1;1;3;// S// =   //51
        2;1;3;3;1;1;// T// =   //52
        2;1;3;1;3;1;// U// =   //53
        3;1;1;1;2;3;// V// =   //54
        3;1;1;3;2;1;// W// =   //55
        3;3;1;1;2;1;// X// =   //56
        3;1;2;1;1;3;// Y// =   //57
        3;1;2;3;1;1;// Z// =   //58
        3;3;2;1;1;1;// [// =   //59
        3;1;4;1;1;1;// \\// =  //60
        2;2;1;4;1;1;// ]// =   //61
        4;3;1;1;1;1;// ^// =   //62
        1;1;1;2;2;4;// _// =   //63
        1;1;1;4;2;2;// NUL//'//  64
        1;2;1;1;2;4;// SOH//a//  65
        1;2;1;4;2;1;// STX//b//  66
        1;4;1;1;2;2;// ETX//c//  67
        1;4;1;2;2;1;// EOT//d//  68
        1;1;2;2;1;4;// ENQ//e//  69
        1;1;2;4;1;2;// ACK//f//  70
        1;2;2;1;1;4;// BEL//g//  71
        1;2;2;4;1;1;// BS// h//  72
        1;4;2;1;1;2;// HT// i//  73
        1;4;2;2;1;1;// LF// j//  74
        2;4;1;2;1;1;// VT// k//  75
        2;2;1;1;1;4;// FF// l//  76
        4;1;3;1;1;1;// CR// m//  77
        2;4;1;1;1;2;// SO// n//  78
        1;3;4;1;1;1;// SI// o//  79
        1;1;1;2;4;2;// DLE//p//  80
        1;2;1;1;4;2;// DC1//q//  81
        1;2;1;2;4;1;// DC2//r//  82
        1;1;4;2;1;2;// DC3//s//  83
        1;2;4;1;1;2;// DC4//t//  84
        1;2;4;2;1;1;// NAK//u//  85
        4;1;1;2;1;2;// SYN//v//  86
        4;2;1;1;1;2;// ETB//w//  87
        4;2;1;2;1;1;// CAN//x//  88
        2;1;2;1;4;1;// EM// y//  89
        2;1;4;1;2;1;// SUB//z//  90
        4;1;2;1;2;1;// ESC//{//  91
        1;1;1;1;4;3;// FS// |//  92
        1;1;1;3;4;1;// GS// }//  93
        1;3;1;1;4;1;// RS// ~//  94
        1;1;4;1;1;3;// US// DEL//95
        1;1;4;3;1;1;//FNC3//FNC3//96
        4;1;1;1;1;3;//FNC2// FNC2//97
        4;1;1;3;1;1;//SHIFT//SHIFT//98
        1;1;3;1;4;1;//CodeC//CodeC//99
        1;1;4;1;3;1;//CodeB//FNC4//CodeB
        3;1;1;1;4;1;//FNC4//CodeA//CodeA
        4;1;1;1;3;1;//FNC1//FNC1//FNC1
        2;1;1;4;1;2;//StartA//StartA//StartA
        2;1;1;2;1;4;//StartB//StartB//StartB
        2;1;1;2;3;2;//StartC//StartC//StartC
        2;3;3;1;1;1;2;//Stop//Stop//Stop'''


class BarTableFactory(object):

    @staticmethod
    def create(code_type):
        if '128' in code_type:
            return BarTable128(code_type)
        elif '39' in code_type:
            return BarTable39('39')
        elif code_type == 'XX':
            return BarTableXX('XX')
        else:
            print('not implemented code type')
            raise Exception


class BarTable:

    def __init__(self, code_type):
        self.code_table = {}    # bscode: char
        self.code_table_sno = {}    # char: serial no
        self.code_table_sed = {}    # char: other number need to use
        self.load_table(code_type)

    def load_table(self, code_type: str):
        raise Exception


class BarTableXX(BarTable):

    def __init__(self, code_type):
        self.code_type_list = ['xx']
        super().__init__(code_type)

    def load_table(self, code_type='XX'):
        if code_type.lower() not in self.code_type_list:
            print('invalid code type:{}'.format(code_type))
            return
        self.code_table = {}


class BarTable128(BarTable):

    def __init__(self, code_type):
        self.code_type_list = ['128a', '128b', '128c']
        super().__init__(code_type)

    def load_table(self, code_type='128c'):
        table_str = BarConstant.table_code128.split('\n')
        for i, rs in enumerate(table_str):
            s = rs.strip().split('//')
            sk = s[0].replace(';', '')
            sa = s[1].strip()   # 128a
            sb = s[2].strip()   # 128b
            sc = s[3].strip()   # 128c
            if i < 64:
                sb = sa
            sd = ''.join([str(int(sk[si]) + int(sk[si + 1])) for si in range(len(sk)-1)])
            self.code_table.update({sk: {'128a': sa, '128b': sb, '128c': sc}.
                                   get(code_type.lower(), sc)})     # not found: 128c
            self.code_table_sno.update({{'128a': sa, '128b': sb, '128c': sc}.
                                        get(code_type.lower(), sc): i})     # not found: 128c
            self.code_table_sed.update({sd: {'128a': sa, '128b': sb, '128c': sc}.
                                        get(code_type.lower(), sc)})    # not found: 128c


class BarTable39(BarTable):

    def __init__(self, code_type):
        self.code_type_list = ['39', '39asc']
        super().__init__(code_type)

        # asc code table
        self.code_table_asc = {'%U': chr(0), '%X': '~', '%Y': 'DEL', '%Z': 'DEL', '%V': '@', '%W': '`'}
        self.code_table_asc.update({'$'+chr(ord('A')+ci): chr(1) for ci in range(26)})
        self.code_table_asc.update({'%A': chr(27), '%B': chr(28), '%C': chr(29),
                                    '%D': chr(30), '%E': chr(31), ' ': ' '})
        self.code_table_asc.update({'/'+chr(ord('A')+ci): chr(33+ci) for ci in range(12)})   # '/A'...'/L'
        self.code_table_asc.update({'.': '-', '/0': '.', '/Z': ':'})
        self.code_table_asc.update({'%'+chr(70+i): chr(59+i) for i in range(5)})     # '%F'...'%J': ';'...'?'
        self.code_table_asc.update({'%'+chr(75+i): chr(91+i) for i in range(5)})     # '%K'...'%O': '['...''
        self.code_table_asc.update({'%'+chr(81+i): chr(123+i) for i in range(4)})     # '%Q'...'%T': '{'...''
        self.code_table_asc.update({'+'+chr(65+i): chr(97+i) for i in range(26)})

    def load_table(self, code_type='39'):
        if code_type.lower() not in self.code_type_list:
            print('invalid code type:{}'.format(code_type))
            return

        # codestr = self.get_codestr().split('\n')
        codestr = BarConstant.table_code39.split('\n')
        for s in codestr:
            sl = s.strip().split(',')
            self.code_table[sl[1]] = sl[0] if len(sl[0]) > 0 else ' '

        sno_dict = {}
        for si in range(10):
            sno_dict[str(si)] = si
        for si in range(26):
            sno_dict[chr(65+si)] = 10+si
        sno_dict['-'] = 36
        sno_dict['.'] = 37
        sno_dict[' '] = 38
        sno_dict['$'] = 39
        sno_dict['/'] = 40
        sno_dict['+'] = 41
        sno_dict['%'] = 42
        self.code_table_sno = sno_dict


class BarDecoderFactory(object):
    """
    create BarDecoder by assigning code_type
    code_type in ['128'] now
    """
    @staticmethod
    def create(code_type: str):
        if code_type.lower() in ['128', '128a', '128b', '128c']:
            return BarDecoder128
        elif code_type.lower() in ['39', '39asc']:
            return BarDecoder39
        else:
            print('not implemented code_type:{}').format(code_type)
            return None


class BarDecoder(object):

    # @abstractclassmethod
    @staticmethod
    def decode(pwlist, code_type):
        print('implemented in subclass!')
        raise Exception

    # @abstractclassmethod
    @staticmethod
    def adjust(codelist_list, code_type):
        raise Exception

    @staticmethod
    def prune(collect_list):
        raise Exception

    @staticmethod
    def fill(codelist, code_type):
        raise Exception

    @staticmethod
    def check(codelist, code_type):
        raise Exception

    @staticmethod
    def compound(codelist, code_type):
        raise Exception


class BarDecoder128(BarDecoder):

    code_type_list = ['128a', '128b', '128c', '128']
    code_table_dict = \
        {'128a': BarTable128('128a').code_table,
         '128b': BarTable128('128b').code_table,
         '128c': BarTable128('128c').code_table,
         '128': BarTable128('128c').code_table,
         }
    code_sno_dict = \
        {'128a': BarTable128('128a').code_table_sno,
         '128b': BarTable128('128b').code_table_sno,
         '128c': BarTable128('128c').code_table_sno,
         '128': BarTable128('128c').code_table_sno,
         }
    esc_sno_dict = \
        {'CodeA': code_sno_dict['128a'],
         'CodeB': code_sno_dict['128b'],
         'CodeC': code_sno_dict['128c']}
    bscode_starta = '211412'
    bscode_startb = '211214'
    bscode_startc = '211232'
    bscode_stop = '2331112'

    @ classmethod
    # code128_decode
    def decode(cls, pwlist, code_type):
        result_list = []

        # type error
        # if code_type not in BarDecoder128.code_type_list:
        if code_type not in cls.code_type_list:
            # print('invalid code type:{}'.format(code_type))
            # raise Exception
            return result_list

        # length error: 128code_len = 6*n + 13
        if (len(pwlist) - 1) % 6 > 0:
            return result_list

        # decode to bscode
        bscode_list = []
        pwlen = len(pwlist)
        for pi in range(0, pwlen, 6):
            if pi + 8 < pwlen:
                ws = pwlist[pi:pi+6]
            else:
                ws = pwlist[pi:]
            sw = sum(ws)
            si = ''
            bar_unit_len = 11 if len(ws) == 6 else 13
            for r in ws:
                bv = round(bar_unit_len * r / sw)
                bv = 1 if bv == 0 else 4 if bv > 4 else bv
                si = si + str(bv)
            bscode_list.append(si)
            if pi+8 >= pwlen:
                break

        # select codeset
        if code_type not in cls.code_table_dict:
            code_type = {cls.bscode_starta: '128a',
                         cls.bscode_startb: '128b',
                         cls.bscode_startc: '128c'}.\
                         get(bscode_list[0], '128c')

        # decode, including mixing encoding among 128a, 128b, 128c
        # current codetype, escape code to new type
        # decode checkcode digit str
        main_set = code_type
        last_set = code_type
        curr_set = code_type
        # code_dict = BarDecoder128.code_table_dict[curr_set]
        code_dict = cls.code_table_dict[curr_set]
        for bi, bs in enumerate(bscode_list):
            # decode check_code specially
            if bi == bscode_list.__len__()-2:
                code_dict = cls.code_table_dict[main_set]
                code_sno = cls.code_sno_dict[main_set]
                dc = '{:03d}'.format(code_sno[code_dict[bs]]) if bs in code_dict else '**'  # error check code
            else:
                dc = code_dict[bs] if bs in code_dict else '**'     # ** not in tables of all barcode type
            result_list.append(dc)
            if dc in ['StartA', 'CodeA']:
                if ((bi == 0) & (dc == 'StartA')) | ((bi > 0) & (dc == 'CodeA')):
                    curr_set = '128a'
            elif dc in ['StartB', 'CodeB']:
                if ((bi == 0) & (dc == 'StartB')) | ((bi > 0) & (dc == 'CodeB')):
                    curr_set = '128b'
            elif dc in ['StartC', 'CodeC']:
                if ((bi == 0) & (dc == 'StartC')) | ((bi > 0) & (dc == 'CodeC')):
                    curr_set = '128c'
            if curr_set != last_set:
                code_dict = cls.code_table_dict[curr_set]
                last_set = curr_set

        return result_list

    @classmethod
    # code128_adjust
    def adjust(cls, codelist_list, code_type):
        """
        128c now
        useful to CodeB Escape code processing
        CodeB+code: if code not in 0-9
        :param codelist_list
        :param code_type
        :return: new codelist_list
        """
        # just for 128c
        # set code='*' after CodeB if not digits, code=0-9 if int-16 in [0,9]
        if code_type == '128c':
            # print('adjusted---')
            result_lists = []
            for cl in codelist_list:
                # if cl[-3] in ['CodeA', 'CodeB', 'CodeC']:
                #    cl.append('Stop')
                tl = cl[0:2]
                for j in range(2, len(cl)):
                    # Escape by CodeB
                    if cl[j-1] == 'CodeB':
                        # if not cl[j].isdigit():
                        #    cl[j] = '**'
                        if cl[j].isdigit():
                            if (len(cl[j]) == 2) & (0 <= int(cl[j])-16 <= 9):
                                cl[j] = str(int(cl[j])-16)
                    # remove repeated check code
                    if j == len(cl)-2:
                        if (len(cl[j]) > 2) & (len(cl[j-1]) > 2):
                            continue
                    tl.append(cl[j])
                result_lists.append(tl)
            return result_lists

        return codelist_list

    @classmethod
    # code128_prune
    def prune(cls, collect_list):

        # invalid collect, not neccessary to prune
        if len(collect_list) < 3:
            return collect_list

        # remove empty element in tail by locating new stop position
        stop_loc = len(collect_list) - 1
        for i in range(len(collect_list)-1, 1, -1):  # enumerate(collect_list[2:], start=2):
            c = collect_list[i]
            # print(i, stop_loc, c)
            if 'Stop' in collect_list[stop_loc]:
                # stop_loc is ok
                # if collect_list[stop_loc]['Stop'] > 30:
                #    continue
                # set i as new stop loc
                if 'Stop' in c:
                    if (c['Stop'] > 20) & (collect_list[stop_loc]['Stop'] < 10):
                        stop_loc = i
                        break
            # tail stop < = 30
            # go back 1 when current count is 0
            if (len(c) == 0) & ('Stop' in collect_list[i - 1]):
                if collect_list[i-1]['Stop'] > 20:   # think the threshold:50
                    stop_loc = i-1
                    break
            # go back 1 when current count less than 10
            if ('Stop' in c) & ('Stop' in collect_list[i - 1]):
                # new loc is higher but now is too little
                if (c['Stop'] < 10) & (collect_list[i - 1]['Stop'] > 20):
                    stop_loc = i - 1
                    break
                # new loc enough high, now is lower
                if (c['Stop'] < collect_list[i - 1]['Stop']) & (collect_list[i-1]['Stop'] > 50):
                    stop_loc = i - 1
                    break
                if (c['Stop'] > 50) & (collect_list[i - 1]['Stop'] > 50):
                    stop_loc = i - 1
                    break

        collect_list = collect_list[0:stop_loc+1]

        # set 'Start', 'Stop' in head/tail code_dict
        meancount = sum([max(list(d.values())) if len(d) > 0 else 0 for d in collect_list])\
            / len(collect_list)
        if 'Start' in ''.join(list(collect_list[0].keys())):
            # set Start code in head code
            collect_list[0] = {k: int(meancount) for k in collect_list[0]
                               if 'Start' in k}
        if 'Stop' in ''.join(list(collect_list[-1].keys())):
            # set stop code in end code
            collect_list[-1] = {'Stop': int(meancount)}

        # prunning redundant node: 'Start', 'Stop' in middle code dict
        collect2 = [collect_list[0]]
        for ci, cd in enumerate(collect_list[1:-1]):  # not head, tail, more than 2 elenments
            # if 0 < ci < len(collect_list)-1:
            if 'Stop' in cd:
                del cd['Stop']
            cdkeys = cd.copy()
            for c0 in cdkeys:
                if 'Start' in c0:
                    del cd[c0]
            collect2.append(cd)
        collect2.append(collect_list[-1])
        # collect_list = collect2

        return collect2
        # _get2x3x1

    @classmethod
    # code128_fill
    def fill(cls, codelist, code_type):
        # need to think 128a, 128b, 128c
        # fill '**' with valid code
        """
        now only used for 128c
        best result is to fill 1 loss code
        ---
        input codelist with loss code as '**'
        output possible codelist_list by filling valid code
        when loss code number more than 1, result codelist may be too much
        :param codelist:  with loss code
        :param code_type:  used code type, as 128a, 128b, 128c
        :return: list of codelist
        """

        # remove head and tail
        if len(codelist) > 3:
            start_code = codelist[0]
            stop_code = codelist[-1]
            codelist = codelist[1:-1]
        else:
            return [codelist]

        loss_dict = {i: 0 for i, s in enumerate(codelist) if s == '**'}
        loss_num = len(loss_dict)
        loss_keys = list(loss_dict.keys())

        # print('no loss in codelist:', loss_dict)
        if loss_num == 0:
            return [codelist]

        # search valid code
        result_codelists = []
        codebnum = ''.join(codelist).count('CodeB*')
        max_sum = 100 ** (loss_num-codebnum) * 10**codebnum
        cur_sum = 0
        while cur_sum < max_sum:
            for i in loss_keys:
                if codelist[i - 1] != 'CodeB':
                    if loss_dict[i] < 99:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                else:  # code_list[i - 1] == 'CodeB':
                    if loss_dict[i] < 10:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                loss_dict[i] = 0
            # check_code
            code_new = []
            for j, c in enumerate(codelist):
                if j not in loss_keys:
                    code_new.append(c)
                elif codelist[j-1] == 'CodeB':
                    code_new.append(str(loss_dict[j]))
                else:
                    if loss_dict[j] < 10:
                        code_new.append('0' + str(loss_dict[j]))
                    else:
                        code_new.append(str(loss_dict[j]))
            if BarDecoderFactory.create(code_type).check([start_code]+code_new+[stop_code], code_type)[0]:
                result_codelists.append([start_code]+code_new+[stop_code])
            cur_sum = cur_sum + 1

        return result_codelists

    @classmethod
    # code128check
    def check(cls, codelist, code_type):
        """
        calculate check result, include in list [check_validity, check_sum, code_checkvalue_list]
        :param codelist: list of barcode
        :param code_type: now, code_type is in ['128a', '128b', '128c']
        :return check_value_list: [True or False, checkvalue, [code_serial_no*index, ...]]
        """
        if type(codelist) != list:
            # print('error1: codelist is not list')
            return [False, -1, [-1]]
        if len(codelist) < 4:
            # print('error2: invalid length({}) codelist'.format(len(codelist)))
            return [False, -2, [-1]]
        if code_type not in cls.code_sno_dict:
            # print('error3: invalid code type, not in {}'.format(self.code_sno_dict.keys()))
            return [False, -3, [-1]]
            # code_type = '128c'
        if not codelist[-2].isdigit():
            # print('error4: check code is digit string')
            return [False, -4, [-1]]

        curr_type = code_type
        last_type = code_type
        bt = cls.code_sno_dict[curr_type]
        ck_value_list = []
        ck_sum = 0
        for ci, cc in enumerate(codelist):
            if not isinstance(cc, str):
                ck_value_list.append('**')
            # stop at check_code in [-2]
            if ci == len(codelist)-2:
                if ck_sum % 103 == int(codelist[-2]):
                    return [True, ck_sum % 103, ck_value_list]
                else:
                    return [False, ck_sum % 103, ck_value_list]

            # get check value for each item
            if curr_type != last_type:
                bt = cls.code_sno_dict[curr_type]
                last_type = curr_type
            if cc in bt:
                ck_value = bt[cc] * (1 if ci == 0 else ci)
            elif ci == 0:  # Start_code == '**'
                ck_value = {'128a': 103, '128b': 104, '128c': 105}.get(code_type, 105)  # default to 128c
            else:
                ck_value = -1

            # get check value
            if ck_value >= 0:
                ck_sum += ck_value
                ck_value_list.append(ck_value)
            else:
                ck_value_list.append(cc)

            # switch code type
            if cc in ['StartA', 'CodeA']:
                curr_type = '128a'
            elif cc in ['StartB', 'CodeB']:
                curr_type = '128b'
            elif cc in ['StartC', 'CodeC']:
                curr_type = '128c'
            # think: when stwitch code is SHIFT, just switch one code after it?
            # no standard file to indentify this
        # err end
        return [False, -5, ck_value_list]

    @classmethod
    # code128_compound
    def compound(cls, codelist, code_type):
        if len(codelist) < 4:
            return ''
        if Counter(codelist).get('**', 0) >= 3:
            return ''

        return ''.join([c for c in codelist[1:-2]
                        if c not in ['CodeA', 'CodeB', 'CodeC', 'SHIFT', 'FNC1', 'FNC2'
                        'FNC3', 'FNC4']])


class BarDecoder39(BarDecoder):

    code_type_list = ['39', '39asc']
    code_start = '*'

    # think: there maybe be some problems in the table
    # 0-->0, 0-->/
    # the table is from https://wenku.baidu.com/view/4299e977c281e53a5902ff36.html?sxts=1526858623968
    code_table = BarTableFactory.create('39').code_table
    code_table_sno = BarTableFactory.create('39').code_table_sno
    code_table_asc = BarTableFactory.create('39').code_table_asc

    @classmethod
    # code39_decode
    def decode(cls, pwlist, code_type):

        pwlen = len(pwlist)
        result_codelist = []

        # type error
        if code_type not in cls.code_type_list:
            # print('invalid code type:{}'.format(code_type))
            # raise Exception
            result_codelist.append('***')
            return result_codelist

        # length error
        if pwlen % 10 != 9:
            result_codelist.append('')
            return result_codelist

        # decode to bscode
        # skip seprate code 'space' by set skip=10 and extract pi:pi+9
        for pi in range(0, pwlen, 10):
            ws = pwlist[pi:pi+9]
            # sw = sum(ws)
            si = ''
            bar_max3 = min(nlargest(3, ws))
            bar_stat = 'bar'
            for r in ws:
                if bar_stat == 'bar':
                    si = si + str('11' if r >= bar_max3 else '1')
                    bar_stat = 'space'
                else:
                    si = si + str('00' if r >= bar_max3 else '0')
                    bar_stat = 'bar'
            result_codelist.append(cls.code_table.get(si, '**'))
        return result_codelist

    @classmethod
    # code39_check
    def check(cls, codelist, code_type):
        return True, 0, []

    @classmethod
    def check_sub(cls, codelist):
        # error1: too short, not including start, datacode, checkcode, endcode
        if len(codelist) < 4:
            return [False, -1, []]
        # error2: no checkcode
        if not codelist[-2].isdigit():
            return [True, -2, []]
        check_value = 0
        check_list = []
        for cc in codelist[1:-2]:
            if cc in cls.code_table_sno:
                cv = cls.code_table_sno[cc]
            else:
                cv = -1
            check_value += cv
            check_list.append(cv)
        check_value = check_value % 43
        check_value = check_value % 10
        check_digit = str(check_value)
        return [check_digit == codelist[-2], check_value, check_list]

    @classmethod
    # code39_prune
    def prune(cls, collect_list):
        return collect_list

    @classmethod
    # code39_adjust
    def adjust(cls, codelist_list, code_type='39'):
        return codelist_list

    @classmethod
    # code39_fill
    def fill(cls, codelist, code_type='39'):
        return codelist

    @classmethod
    # code39_compound
    def compound(cls, codelist, code_type):
        if len(codelist) < 3:
            return '**'
        result_code = ''
        if code_type == '39asc':
            esc_code = ''
            for dc in codelist[1:-1]:
                if dc in ['%', '/', '+', '$']:
                    esc_code = dc
                    continue
                if esc_code == '':
                    result_code += dc
                else:
                    result_code += cls.code_table_asc.get(esc_code+dc, '')
                    esc_code = ''
        else:
            result_code = ''.join(codelist[1:-1])

        return result_code


class BarDecoderXX(BarDecoder):

    code_type_list = ['XX', 'XXa']
    code_start = 'Start'

    code_table = BarTableFactory.create('XX').code_table
    code_table_sno = BarTableFactory.create('XX').code_table_sno

    @classmethod
    # codeXX_decode
    def decode(cls, pwlist, code_type):
        result_codelist = []
        return result_codelist

    @classmethod
    # codeXX_check
    def check(cls, codelist, code_type='39'):
        return True

    @classmethod
    # codeXX_prune
    def prune(cls, collect_list):
        return collect_list

    @classmethod
    # codeXX_adjust
    def adjust(cls, codelist_list, code_type='39'):
        return codelist_list

    @classmethod
    # codeXX_fill
    def fill(cls, codelist, code_type='39'):
        return codelist


# --- some useful functions in omrmodel or outside
class BarUtil(object):

    @staticmethod
    def show_image(path_file):
        if os.path.isfile(path_file):
            plt.imshow(plt.imread(path_file))
            plt.title(path_file)
            plt.show()
        else:
            print('file \"%s\" is not found!' % path_file)

    @staticmethod
    def find_file_from_pathfile(path_file):
        return path_file.replace('/', '\\').split('\\')[-1]

    @staticmethod
    def find_path_from_pathfile(path_file):
        ts = BarUtil.find_file_from_pathfile(path_file)
        return path_file.replace(ts, '').replace('\\', '/')

    @staticmethod
    def glob_files_from_path(path, substr=''):
        if not os.path.isdir(path):
            return ['']
        file_list = []
        for f in glob.glob(path+'/*'):
            # print(f)
            if os.path.isfile(f):
                if len(substr) == 0:
                    file_list.append(f)
                elif substr in f:
                    file_list.append(f)
            if os.path.isdir(f):
                [file_list.append(s)
                 for s in BarUtil.glob_files_from_path(f, substr)]
        return file_list

    @staticmethod
    def matrix_row_reverse(matrix_2d):
        return matrix_2d[::-1]

    @staticmethod
    def matrix_col_reverse(matrix_2d):
        return np.array([matrix_2d[r, :][::-1] for r in range(matrix_2d.shape[0])])

    @staticmethod
    def matrix_rotate90_right(matrix_2d):
        # matrix_2d[:] = np.array(map(list, zip(*matrix_2d[::-1])))
        temp = map(list, zip(*matrix_2d[::-1]))
        return np.array(list(temp))

    @staticmethod
    def find_high_count_element(mylist: list):
        cn = Counter(mylist)
        if len(cn) > 0:
            return cn.most_common(1)[0][0]
        else:
            return 0

    @staticmethod
    def find_high_count_continue_element(mylist: list):
        if len(mylist) == 0:
            print('empty list')
            return -1
        countlist = [0 for _ in mylist]
        for i, e in enumerate(mylist):
            for ee in mylist[i:]:
                if ee == e:
                    countlist[i] += 1
                else:
                    break
        m = max(countlist)
        p = countlist.index(m)
        return mylist[p]

    @staticmethod
    def softmax(vector):
        sumvalue = sum([np.exp(v) for v in vector])
        return [np.exp(v)/sumvalue for v in vector]

    @staticmethod
    def image_slant_line(img, line_no, angle):
        if (line_no < 0) or (line_no >= img.shape[0]):
            print('invalid line_no:{}'.format(line_no))
            return
        img0 = []
        last = img[line_no, 0]
        for p in range(img.shape[1]):
            slant = line_no + int(np.tan(angle*3.14156/180)) * (p - img.shape[1])/2
            slant = int(slant)
            if 0 < slant < img.shape[0]:
                img0.append(img[slant, p])
                last = img[slant, p]
            else:
                img0.append(last)
        return img0

    @staticmethod
    def image_resize(img, reshape=(100, 200)):
        return cv2.resize(img, reshape)

    @staticmethod
    def image_amplify(img, ratio_row, ratio_col):
        return cv2.resize(img, (int(img.shape[1]*ratio_col), int(img.shape[0]*ratio_row)))

    @staticmethod
    def check_repeat_in_fields(df: pd.DataFrame, field_list):
        """
        check record repeated times on a field
        field_type must be str
        :param df: input dataframe
        :param field_list: fields to checked
        :return: dataframe to describe field repeated times count
        """
        result_df = pd.DataFrame({'field': [], 'count': []})
        for f in field_list:
            if f in df.columns:
                gf = df.groupby(f)[f].count()
                gf = gf[gf > 1]
                rp = pd.DataFrame({'field': [fd for fd in gf.index],
                                   'count': [int(fc) for fc in gf.values]})
                result_df = result_df.append(rp)
        result_df.loc[:, 'count'] = result_df['count'].astype(int)
        return result_df