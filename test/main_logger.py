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
                 filename='stm.log',
                 level='info',
                 when='D',
                 back_count=3,
                 ):

        # stat
        self.logging_file = True
        self.logging_consol = True

        # para
        self.filename = filename
        self.level = level
        self.when = when
        self.back_count = 3 if back_count is not int else back_count
        self.format = '   %(message)s'
        self.when = when
        # self.format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        # set logger
        self.logger = logging.getLogger(self.filename)              # file name
        self.logger.setLevel(self.level_relations.get(self.level))  # 设置日志级别
        self.logger_format = logging.Formatter(self.format)         # 设置日志格式

        # set handlers
        self.stream_handler = None
        self.rotating_file_handler = None
        self.set_handlers(self.logger_format)

    def loginfo(self, ms=''):
        self.logger.handlers = []
        if self.logging_consol:
            self.logger.addHandler(self.stream_handler)
        if self.logging_file:
            self.logger.addHandler(self.rotating_file_handler)
        self.logger.info(ms)
        self.logger.handlers = []

    def loginfo_start(self, ms=''):
        first_logger_format = logging.Formatter('='*100 + '\n[%(message)s]  at [%(asctime)s]\n ' + '-'*100)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        self.set_handlers(self.logger_format)

    def loginfo_end(self, ms=''):
        first_logger_format = logging.Formatter('-'*100 + '\n[%(message)s]  at [%(asctime)s]\n ' + '='*100)
        self.set_handlers(first_logger_format)
        self.loginfo(ms)
        self.set_handlers(self.logger_format)

    def set_handlers(self, log_format):
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(log_format)
        self.rotating_file_handler = handlers.TimedRotatingFileHandler(
                    filename=self.filename,
                    when=self.when,
                    backupCount=self.back_count,
                    encoding='utf-8'
                )
        self.rotating_file_handler.setFormatter(log_format)
