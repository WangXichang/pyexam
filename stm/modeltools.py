# coding: utf-8


import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as sts
from stm import modelconfig as mcf


class ModelTools:

    @classmethod
    def plot_models(cls, font_size=12, hainan='900'):
        _names = ['shanghai', 'zhejiang', 'beijing', 'tianjin', 'shandong', 'guangdong', 'ss7', 'hn900']
        if hainan == '300':
            _names.remove('hn900')
            _names.append('hn300')
        elif hainan is None:
            _names.remove('hn900')
        ms_dict = dict()
        for _name in _names:
            ms_dict.update({_name: ModelTools.get_model_describe(name=_name)})

        plot.figure('New Gaokao Score mcf.Models: name(mean, std, skewness)')
        plot.rcParams.update({'font.size': font_size})
        for i, k in enumerate(_names):
            plot.subplot(240+i+1)
            _wid = 2
            if k in ['shanghai']:
                x_data = range(40, 71, 3)
            elif k in ['zhejiang', 'beijing', 'tianjin']:
                x_data = range(40, 101, 3)
            elif k in ['shandong']:
                x_data = [x for x in range(26, 100, 10)]
                _wid = 8
            elif k in ['guangdong']:
                x_data = [np.mean(x) for x in mcf.Models[k].section][::-1]
                _wid = 10
            elif k in ['ss7']:
                x_data = [np.mean(x) for x in mcf.Models[k].section][::-1]
                _wid = 10
            elif k in ['hn900']:
                x_data = [x for x in range(100, 901)]
                _wid = 1
            elif k in ['hn300']:
                x_data = [x for x in range(60, 301)]
                _wid = 1
            else:
                raise ValueError(k)
            plot.bar(x_data, mcf.Models[k].ratio[::-1], width=_wid)
            plot.title(k+'({:.2f}, {:.2f}, {:.2f})'.format(*ms_dict[k]))

    @classmethod
    def add_model(cls, model_type='plt', name=None, ratio_list=None, section_list=None, desc=''):
        if model_type not in ['ppt', 'plt']:
            print('error model type={}, valid type:{}'.format(model_type, ['ppt', 'plt']))
            return
        if name in mcf.Models:
            print('name existed in current mcf.Models_dict!')
            return
        if len(ratio_list) != len(section_list):
            print('ratio is not same as segment !')
            return
        for s in section_list:
            if len(s) > 2:
                print('segment is not 2 endpoints: {}-{}'.format(s[0], s[1]))
                return
            if s[0] < s[1]:
                print('the order is from large to small: {}-{}'.format(s[0], s[1]))
                return
        if not all([s1 >= s2 for s1, s2 in zip(section_list[:-1], section_list[1:])]):
            print('section endpoints order is not from large to small!')
            return
        mcf.Models.update({name: mcf.ModelFields(
                                                 model_type,
                                                 ratio_list,
                                                 section_list,
                                                 desc)})

    @classmethod
    def show_models(cls):
        for k in mcf.Models:
            v = mcf.Models[k]
            print('{:<20s} {},  {} '.format(k, v.type, v.desc))
            print('{:<20s} {}'.format(' ', v.ratio))
            print('{:<20s} {}'.format('', v.section))

    @classmethod
    def get_model_describe(cls, name='shandong'):
        __ratio = mcf.Models[name].ratio
        __section = mcf.Models[name].section
        if name == 'hn900':
            __mean, __std, __skewness = 500, 100, 0
        elif name == 'hn300':
            __mean, __std, __skewness = 180, 30, 0
        else:
            samples = []
            [samples.extend([np.mean(s)]*int(__ratio[i])) for i, s in enumerate(__section)]
            __mean, __std, __skewness = np.mean(samples), np.std(samples), sts.skew(np.array(samples))
        return __mean, __std, __skewness
