# encoding: utf-8


import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', back_count=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):

        # set logger
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        # 向控制台输出日志
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format_str)
        self.logger.addHandler(stream_handler)

        # 日志按文件大小写入文件
        # 1MB = 1024 * 1024 bytes
        # 这里设置文件的大小为500MB
        rotating_file_handler = handlers.RotatingFileHandler(
            filename=filename, mode='a', maxBytes=1024 * 1024 * 500, backupCount=back_count, encoding='utf-8')
        rotating_file_handler.setFormatter(format_str)
        self.logger.addHandler(rotating_file_handler)


log = Logger('stm.log', level='info')

log.logger.info('[test] stm-logging')
