# encoding: utf-8


import os
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug':    logging.DEBUG,
        'info':     logging.INFO,
        'warn':     logging.WARNING,
        'error':    logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
                 filename,
                 level='info',
                 when='D',
                 back_count=3,
                 ):

        # check valid
        self.check_filename(filename)
        self.check_level(level)

        # para
        self.filename = filename
        self.level = level
        self.when = when
        self.back_count = 3 if back_count is not int else back_count
        self.format = '[%(asctime)s] <%(levelname)s>:  %(message)s'
        self.when = when
        # self.format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        # set logger
        self.logger = logging.getLogger(self.filename)              # file name
        self.logger_format = logging.Formatter(self.format)         # 设置日志格式
        self.logger.setLevel(self.level_relations.get(self.level))  # 设置日志级别

    # 向控制台输出日志
    def set_consol_logger(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.logger_format)
        self.logger.addHandler(stream_handler)

    # 按天写入文件
    def set_file_day_logger(self):
        rotating_file_handler = handlers.TimedRotatingFileHandler(
            filename=self.filename,
            when=self.when,
            backupCount=self.back_count,
            encoding='utf-8'
            )
        # maxBytes=1024 * 1024 * 500,
        rotating_file_handler.setFormatter(self.logger_format)
        self.logger.addHandler(rotating_file_handler)

    def check_filename(self, filename):
        path = None
        if not isinstance(filename, str):
            print(f'filename={filename} is not str!')
            return False
        elif (len(filename.split('\\')) == 1) or (len(filename.split('/')) == 1):
            path = './'
        else:
            filename = filename.replace('\\', '/')
            path = '/'.join(filename.split('/')[:-1]) + '/'
        if not os.path.isdir(path):
            print('error dir in filename={}'.format(filename))
            return False
        return True

    def check_level(self, level=None):
        if not isinstance(level, str):
            print('level is not str!')
            return False
        elif level not in self.level_relations.keys():
            print('level error: not in {}'.format(list(self.level_relations.keys())))


def loginfo(logger=None, ms=''):
    if logger:
        logger.logger.info(ms)
        return True
    else:
        print(ms)
    return False

#
# log = Logger('stm.log', level='info')
#
# log.logger.info('\n   [test] stm-logging')
