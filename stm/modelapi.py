# coding: utf-8


from stm import modelconfig as mcf
from  collections import namedtuple
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as sts


def round45r(number, digits=0):
    int_len = len(str(int(abs(number))))
    if int_len + abs(digits) <= 16:
        err_ = (1 if number >= 0 else -1)*10**-(16-int_len)
        if digits > 0:
            return round(number + err_, digits) + err_
        else:
            return int(round(number + err_, digits))
    else:
        raise NotImplemented


def use_ellipsis(digit_seq):
    _digit_seq = None
    if type(digit_seq) == str:
        _digit_seq = tuple(int(x) for x in digit_seq)
    elif type(digit_seq) in (list, tuple):
        _digit_seq = digit_seq
    ellipsis_list = []
    if len(_digit_seq) > 0:
        start_p, end_p, count_p = -1, -1, -1
        for p in _digit_seq:
            if p == _digit_seq[0]:
                start_p, end_p, count_p = p, p, 1
            if p == _digit_seq[-1]:
                if count_p == 1:
                    ellipsis_list += [start_p, p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p, p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p, p]
                elif p == end_p + 1:
                    ellipsis_list += [start_p, Ellipsis, p]
                else:
                    ellipsis_list += [start_p, Ellipsis, end_p, p]
                break
            if p > end_p + 1:
                if count_p == 1:
                    ellipsis_list += [start_p]
                elif count_p == 2:
                    ellipsis_list += [start_p, end_p]
                elif count_p == 3:
                    ellipsis_list += [start_p, end_p-1, end_p]
                else:
                    ellipsis_list += [start_p, Ellipsis, end_p]
                count_p = 1
                start_p, end_p = p, p
            elif p == end_p + 1:
                end_p, count_p = p, count_p + 1
    return str(ellipsis_list).replace('Ellipsis', '...')


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

        plot.figure('New Gaokao Score Models: name(mean, std, skewness)')
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
        if model_type not in mcf.MODEL_TYPE:
            print('error model type={}, valid type:{}'.format(model_type, mcf.MODEL_TYPE))
            return
        if name in mcf.Models:
            print('name existed in current models_dict!')
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
        mcf.Models.update({name: mcf.ModelFields(model_type,
                                                 ratio_list,
                                                 section_list,
                                                 desc)})
        MODELS_NAME_LIST = mcf.Models.keys()

    @classmethod
    def show_models(cls):
        for k in mcf.Models:
            v = mcf.Models[k]
            print('{:<20s} {}'.format(k, v.ratio))
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

    @classmethod
    def get_section_pdf(cls,
            start=21,
            end=100,
            section_num=8,
            std_num=2.6,
            add_cut_error=True,
            ):
        """
        # get pdf, cdf, cutoff_err form section end points,
        # set first and end seg to tail ratio from norm table
        # can be used to test std from seg ratio table
        # for example,
        #   get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.9508)
        #   [0.03000, 0.07513, 0.16036, 0.234265, ..., 0.03000],
        #   get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.6065)
        #   [0.02729, 0.07272, 0.16083, 0.23916, ..., 0.02729],
        #   it means that std==15.95 is fitting ratio 0.03,0.07 in the table

        :param start:   start value
        :param end:     end value
        :param section_num: section number
        :param std_num:     length from 0 to max equal to std_num*std, i.e. std = (end-start)/2/std_num
        :return: dict{'val': (), 'pdf': (), 'cdf': (), 'cut_error': float, 'add_cut_error': bool}
        """
        _mean, _std = (end+start)/2, (end-start)/2/std_num
        section_point_list = np.linspace(start, end, section_num+1)
        cut_error = sts.norm.cdf((start-_mean)/_std)
        result = dict()
        pdf_table = [0]
        cdf_table = [0]
        last_pos = (start-_mean)/_std
        _cdf = 0
        for i, pos in enumerate(section_point_list[1:]):
            _zvalue = (pos-_mean)/_std
            this_section_pdf = sts.norm.cdf(_zvalue)-sts.norm.cdf(last_pos)
            if (i == 0) and add_cut_error:
                this_section_pdf += cut_error
            pdf_table.append(this_section_pdf)
            cdf_table.append(this_section_pdf + _cdf)
            last_pos = _zvalue
            _cdf += this_section_pdf
        if add_cut_error:
            pdf_table[-1] += cut_error
            cdf_table[-1] = 1
        result.update({'val': tuple(section_point_list)})
        result.update({'pdf': tuple(pdf_table)})
        result.update({'cdf': tuple(cdf_table)})
        result.update({'cut_error': cut_error})
        result.update({'add_cut_error': add_cut_error})
        return result

    # single ratio-seg search in seg-percent sequence
    @classmethod
    def get_seg_from_seg_ratio_sequence(cls,
                                        dest_ratio,
                                        seg_seq,
                                        ratio_seq,
                                        tiny_value=10**-8,
                                        ):
        # comments:
        #   if value type is Fraction in ratio_sequence,
        #   use limit_denominator for dest_ratio or str type, Fraction(str)
        #   because of the Fraction number error in pandas.field(Fraction)
        #   can't compare ratio with Fraction number in pd.DataFrame directly
        #   for example: dest_ratio = fr.Fraction(ratio).limit_denominator(10**8)

        Result = namedtuple('Result',
                            ['this_seg_near', 'top', 'bottom',
                             'this_seg', 'last_seg',
                             'this_percent', 'last_percent',
                             'dist_to_this', 'dist_to_last'])
        # too big dest_ratio
        if dest_ratio > list(ratio_seq)[-1]:
            result = Result(True, False, True,
                            list(seg_seq)[-1], list(seg_seq)[-1],
                            list(ratio_seq)[-1], list(ratio_seq)[-1],
                            -1, -1
                            )
            return result
        last_percent = -1
        last_seg = -1
        _top, _bottom, _len = False, False, len(seg_seq)
        for row_id, (seg, percent) in enumerate(zip(seg_seq, ratio_seq)):
            this_percent = percent
            this_seg = seg
            if row_id == _len:
                _bottom = True
            # meet a percent that bigger or at table bottom
            if (this_percent >= dest_ratio) or _bottom:
                if row_id == 0:
                    _top = True
                dist_to_this = float(this_percent - dest_ratio)
                dist_to_last = float(dest_ratio - last_percent)
                if _top:    # at top and percent >= ratio
                    dist_to_this = float(this_percent - dest_ratio)
                if (this_percent - dest_ratio) < tiny_value:  # equal to ratio
                    dist_to_this = 0
                this_seg_near = False if dist_to_last < dist_to_this else True
                return Result(this_seg_near, _top, _bottom,
                              this_seg, last_seg,
                              float(this_percent), float(last_percent),
                              float(dist_to_this), float(dist_to_last),
                              )
            last_percent = this_percent
            last_seg = this_seg
        return Result(False, False, False, -1, -1, -1, -1, -1, -1)

    @classmethod
    def get_raw_section(cls,
            section_ratio_cumu_sequence,
            raw_score_sequence,
            raw_score_percent_sequence,
            mode_ratio_prox='upper_min',
            mode_ratio_cumu='no',
            mode_sort_order='d',
            mode_section_startpoint_first='real_max_min',
            mode_section_startpoint_else='step_1',
            mode_section_lost='ignore',
            tiny_value=10**-12,
            ):
        """
        section point searching in seg-percent sequence
        warning: lost section if else start percent is larger than dest_ratio to locate
        :param section_ratio_cumu_sequence: ratio for each section
        :param raw_score_sequence:   score point corresponding to ratio_sequence
        :param raw_score_percent_sequence: real score cumulative percent
        :param mode_ratio_prox: 'upper_min, lower_max, near_max, near_min'
        :param mode_ratio_cumu: 'yes', 'no', if 'yes', use cumulative ratio in searching process
        :param mode_section_startpoint_first: how to choose first point on section
        :param tiny_value: if difference between ratio and percent, regard as equating
        :return: section_list, section_point_list, section_real_ratio_list
        """
        # step-0: check seg_sequence order
        _order = [(x > y) if mode_sort_order in ['d', 'descending'] else (x < y)
                  for x, y in zip(raw_score_sequence[:-1], raw_score_sequence[1:])]
        if not all(_order):
            print('seg sequence is not correct order:{}'.format(mode_sort_order))
            raise ValueError
        if len(raw_score_sequence) != len(raw_score_percent_sequence):
            print('seg_sequence and ratio_sequence are not same length!')
            raise ValueError

        # step-1: locate first section point
        _startpoint = -1
        for seg, ratio in zip(raw_score_sequence, raw_score_percent_sequence):
            if mode_section_startpoint_first == 'real_max_min':
                if ratio < tiny_value:
                    # skip empty seg
                    continue
                else:
                    # first non-empty seg
                    _startpoint = seg
                    break
            else:
                # choose first seg if 'defined_max_min'
                _startpoint = seg
                break
        section_point_list = [_startpoint]

        # step-2: set end-point of else sections
        section_percent_list = []
        dest_ratio = None
        last_ratio = 0
        real_percent = 0
        goto_bottom = False
        for ratio in section_ratio_cumu_sequence:
            if dest_ratio is None:
                dest_ratio = ratio
            else:
                if mode_ratio_cumu == 'yes':
                    dest_ratio = real_percent + ratio - last_ratio
                else:
                    dest_ratio = ratio
            # avoid to locate invalid ratio
            _seg, _percent = -1, -1
            # if real_percent > dest_ratio:
            #     # this section is lost in last section
            #     # set to (-1, -1)
            #     pass
            if not goto_bottom:
                result = ModelTools.get_seg_from_seg_ratio_sequence(
                    dest_ratio,
                    raw_score_sequence,
                    raw_score_percent_sequence,
                    tiny_value)
                # strategy: mode_ratio_prox:
                # equal to this or choosing upper min
                if (result.dist_to_this < tiny_value) or (mode_ratio_prox == 'upper_min'):
                    _seg, _percent = result.this_seg, result.this_percent
                # equal to last or choosing lower max
                elif (mode_ratio_prox == 'lower_max') or (result.dist_to_last < tiny_value):
                    _seg, _percent = result.last_seg, result.last_percent
                # near or 'near_max' or 'near_min'
                elif 'near' in mode_ratio_prox:
                    if result.dist_to_this < result.dist_to_last:
                        _seg, _percent = result.this_seg, result.this_percent
                    elif result.dist_to_this > result.dist_to_last:
                        _seg, _percent = result.last_seg, result.last_percent
                    else: # dist is same
                        if mode_ratio_prox == 'near_max':
                            _seg, _percent = result.this_seg, result.this_percent
                        else:
                            _seg, _percent = result.last_seg, result.last_percent
                else:
                    print('mode_ratio_prox error: {}'.format(mode_ratio_prox))
                    raise ValueError
            # avoid to repeat search if dest_ratio > 1 last time
            if dest_ratio > 1:
                goto_bottom = True
            section_point_list.append(_seg)
            section_percent_list.append(float(_percent))
            last_ratio = ratio
            if _percent > 0:    # jump over lost section
                real_percent = _percent

        # step-3: process same point
        #         that means a lost section
        new_section = [section_point_list[0]]
        _step = -1 if mode_sort_order in ['d', 'descending'] else 1
        for p, x in enumerate(section_point_list[1:]):
            if x != section_point_list[p]:
                new_section.append(x)
            else:
                if mode_section_lost == 'ignore':
                    # new_section.append(-1)
                    pass
                elif mode_section_lost == 'next_one_point':
                    new_section.append(x+_step)
                elif mode_section_lost == 'next_two_point':
                    # maybe coliide to next section if it is single point section
                    new_section.append(x+2*_step)
        section_point_list = new_section

        # new_percent = [section_percent_list[0]]
        # _ = [new_percent.append(x) for i, x in enumerate(section_percent_list[1:]) if x != section_percent_list[i]]
        # section_percent_list = new_percent

        # step-4: make section
        #   with strategy: mode_section_startpoint_else
        #                  default: step_1
        section_list = [(x-1, y) if i > 0 else (x, y)
                        for i, (x, y) in enumerate(zip(section_point_list[0:-1], section_point_list[1:]))]

        # depricated: it is unreasonable to jump empty point, 
        #   when add some real score to empty point, no value can be used
        # if mode_section_startpoint_else == 'jump_empty':
        #     new_step = 0
        #     new_section_endpoints = []
        #     for x, y in section_list:
        #         while (x in raw_score_sequence) and (x != y):
        #             pos = list(raw_score_sequence).index(x)
        #             if raw_score_percent_sequence[pos] == raw_score_percent_sequence[pos - 1]:
        #                 new_step += -1 if x > y else 1
        #             else:
        #                 break
        #         new_section_endpoints.append((x + new_step, y))
        #     section_list = new_section_endpoints

        if mode_section_startpoint_else == 'share':
            section_list = [(x, y) for i, (x, y)
                            in enumerate(zip(section_point_list[0:-1], section_point_list[1:]))]

        # step-5: add lost section with (-1, -1)
        less_len = len(section_ratio_cumu_sequence) - len(section_list)
        if less_len > 0:
            section_list += [(-1, -1)] * less_len
            section_percent_list += [-1] * less_len

        Result = namedtuple('result', ['section', 'point', 'percent'])
        return Result(section_list, section_point_list, section_percent_list)

    @classmethod
    def get_plt_formula(cls,
            raw_section,
            out_section,
            mode_section_degraded='map_to_max',
            ):
        plt_formula = dict()
        i = 0
        for rsec, osec in zip(raw_section, out_section):
            # rsec is degraded
            if rsec[0] == rsec[1]:
                a = 0
                if mode_section_degraded == 'map_to_max':
                    b = max(osec)
                elif mode_section_degraded == 'map_to_min':
                    b = min(osec)
                elif mode_section_degraded == 'map_to_mean':
                    b = np.mean(osec)
                else:
                    raise ValueError
            else:
                y2, y1 = osec[1], osec[0]
                x2, x1 = rsec[1], rsec[0]
                a = (y2 - y1) / (x2 - x1)
                b = (y1 * x2 - y2 * x1) / (x2 - x1)
            plt_formula.update({i: ((a, b),
                                    rsec,
                                    osec,
                                    'y = {:.8f}*x + {:.8f}'.format(a, b),
                                    )
                                })
            i += 1

        # function of formula
        def formula(x):
            for k in plt_formula:
                if plt_formula[k][1][0] <= x <= plt_formula[k][1][1]:
                    return plt_formula[k][0][0] * x + plt_formula[k][0][1]
            return -1

        Result = namedtuple('Result', ('formula', 'coeff_raw_out_section_formula'))
        return Result(formula, plt_formula)

    @classmethod
    def get_ppt_formula(cls,
                        raw_score_points,
                        raw_score_percent,
                        out_score_points,
                        out_score_percent,
                        mode_ratio_prox='upper_min',
                        mode_ratio_cumu='no',
                        mode_sort_order='d',
                        mode_raw_score_max='map_by_ratio',
                        mode_raw_score_min='map_by_ratio',
                        tiny_value=10**-12
                        ):
        ppt_formula = dict()
        _rmax, _rmin = max(raw_score_points), min(raw_score_points)
        if mode_sort_order in ['d', 'descending']:
            if any([x <= y for x,y in zip(raw_score_points[:-1], raw_score_points[1:])]):
                print('raw score sequence is not correct order: {}'.format(mode_sort_order))
                return
        # lcoate out-score to raw-ratio in out-score-ratio-sequence
        dest_ratio = None
        last_ratio = 0
        real_percent = 0
        for rscore, raw_ratio in zip(raw_score_points, raw_score_percent):
            if rscore == _rmax:
                if mode_raw_score_max == 'map_to_max':
                    ppt_formula.update({rscore: max(out_score_points)})
                    continue
            if rscore == _rmin:
                if mode_raw_score_min == 'map_to_min':
                    ppt_formula.update({rscore: min(out_score_points)})
                    continue

            if dest_ratio is None:
                dest_ratio = raw_ratio
            else:
                if mode_ratio_cumu == 'yes':
                    dest_ratio = real_percent + raw_ratio - last_ratio
                else:
                    dest_ratio = raw_ratio

            # set invalid ratio if can not found ration in out_percent
            _seg, _percent = -1, -1
            result = ModelTools.get_seg_from_seg_ratio_sequence(
                dest_ratio,
                out_score_points,
                out_score_percent,
                tiny_value)
            # print(raw_ratio, result)

            # strategy: mode_ratio_prox:
            # choose this_seg if near equal to this or upper_min
            if (result.dist_to_this < tiny_value) or (mode_ratio_prox == 'upper_min'):
                _seg, _percent = result.this_seg, result.this_percent
            # choose last if last is near or equal
            elif (mode_ratio_prox == 'lower_max') or (result.dist_to_last < tiny_value):
                _seg, _percent = result.last_seg, result.last_percent
            elif 'near' in mode_ratio_prox:
                if result.dist_to_this < result.dist_to_last:
                    _seg, _percent = result.this_seg, result.this_percent
                elif result.dist_to_this > result.dist_to_last:
                    _seg, _percent = result.last_seg, result.last_percent
                else:
                    if mode_ratio_prox == 'near_max':
                        _seg, _percent = result.this_seg, result.this_percent
                    else:
                        _seg, _percent = result.last_seg, result.last_percent
            else:
                print('mode_ratio_prox error: {}'.format(mode_ratio_prox))
                raise ValueError
            ppt_formula.update({rscore: (_seg, _percent)})

        # function of formula
        def formula(x):
            if x in ppt_formula:
                return ppt_formula[x]
            else:
                return -1

        Result = namedtuple('Result', ('formula', 'map_dict'))
        return formula, ppt_formula
