# coding: utf-8


import time
from  collections import namedtuple
import numpy as np
import pandas as pd
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


def run_timer(fun):

    def dec_fun(*args, **kwargs):
        st = time.time()
        print('process start: {}'.format(fun))
        result = fun(*args, **kwargs)
        print('process[{}] elapsed time: {:.3f}'.format(fun, time.time() - st))
        return result

    return dec_fun


def use_ellipsis_in_digits_seq(digit_seq):
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


class ModelAlgorithm:

    @classmethod
    def get_section_pdf(cls,
                        start=21,
                        end=100,
                        section_num=8,
                        std_num=2.6,
                        add_cutoff=True,
                        model_type='plt',
                        ratio_coeff=1,      #1, or 100
                        sort_order='d',
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
        :param add_cutoff: bool, if adding cutoff cdf() at edge point
                           i.e. cdf(-std_num), cdf(-4) = 3.167124183311986e-05, cdf(-2.5098)=0.029894254950869625
        :param model_type: str, 'plt' or 'ppt'
        :return: namedtuple('result', ('section':((),...), 'pdf': (), 'cdf': (), 'cutoff': float, 'add_cutoff': bool))
        """
        _mean, _std = (end+start)/2, (end-start)/2/std_num
        section_point_list = np.linspace(start, end, section_num+1)
        cutoff = sts.norm.cdf((start-_mean)/_std)
        pdf_table = [0]
        cdf_table = [0]
        last_pos = (start-_mean)/_std
        _cdf = 0
        for i, pos in enumerate(section_point_list[1:]):
            _zvalue = (pos-_mean)/_std
            this_section_pdf = sts.norm.cdf(_zvalue)-sts.norm.cdf(last_pos)
            if (i == 0) and add_cutoff:
                this_section_pdf += cutoff
            pdf_table.append(this_section_pdf)
            cdf_table.append(this_section_pdf + _cdf)
            last_pos = _zvalue
            _cdf += this_section_pdf
        if add_cutoff:
            pdf_table[-1] += cutoff
            cdf_table[-1] = 1
        if model_type == 'plt':
            section_list = [(x, y) if i == 0 else (x+1, y)
                            for i, (x, y) in enumerate(zip(section_point_list[:-1], section_point_list[1:]))]
        else:
            section_list = [(x, x) for x in section_point_list]
        if ratio_coeff != 1:
            pdf_table = [x*ratio_coeff for x in pdf_table]
        if sort_order in ['d', 'descending']:
            section_list = sorted(section_list, key=(lambda x: -x[0]))
        result = namedtuple('Result', ('section', 'pdf', 'cdf', 'point', 'cutoff', 'add_cutoff'))
        r = result(tuple(section_list),
                   tuple(pdf_table),
                   tuple(cdf_table),
                   section_point_list,
                   cutoff,
                   add_cutoff)
        return r

    # find seg in seg-percent sequence by ratio-percent
    @classmethod
    def get_score_from_score_ratio_sequence(
            cls,
            dest_ratio,
            seg_seq,
            ratio_seq,
            tiny_value=10**-12,
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
            mode_section_point_first='real',
            mode_section_point_start='step',
            mode_section_point_last='real',
            mode_section_lost='ignore',
            raw_score_max=100,
            raw_score_min=0,
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
        :param mode_section_point_first: how to choose first point on section
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
            if mode_section_point_first == 'real':
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
            if not goto_bottom:
                result = ModelAlgorithm.get_score_from_score_ratio_sequence(
                    dest_ratio,
                    raw_score_sequence,
                    raw_score_percent_sequence,
                    tiny_value)
                # strategy: mode_ratio_prox:
                # at top and single point
                if result.last_seg < 0:
                    _seg, _percent = result.this_seg, result.this_percent
                # equal to this or choosing upper min
                elif (result.dist_to_this < tiny_value) or (mode_ratio_prox == 'upper_min'):
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
        #         that means a lost section ???!
        _step = -1 if mode_sort_order in ['d', 'descending'] else 1
        new_section = [section_point_list[0]]
        for p, x in enumerate(section_point_list[1:]):
            # if p == 0, the first section is degraded, not lost, because of no section to be lost in
            if (p == 0) or (x != section_point_list[p]):
                # not same as the last
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

        # step-3-2: process last point
        if mode_section_point_last == 'defined':
            _last_value = raw_score_min if mode_sort_order in ['d', 'descending'] else \
                          raw_score_max
            i = 0
            while i < len(section_point_list)-1:
                if section_point_list[-1-i] > 0:
                    if section_point_list[-1-i] != _last_value:
                        section_point_list[-1-i] = _last_value
                        break
                i += 1

        # step-4: make section
        #   with strategy: mode_section_point_start
        #                  default: step
        if mode_section_point_start == 'share':
            section_list = [(x, y) for i, (x, y)
                            in enumerate(zip(section_point_list[0:-1], section_point_list[1:]))]
        else:
            section_list = [(x+_step, y) if i > 0 else (x, y)
                            for i, (x, y) in enumerate(zip(section_point_list[0:-1], section_point_list[1:]))]

        # depricated: it is unreasonable to jump empty point, 
        #   when add some real score to empty point, no value can be used
        # if mode_section_point_start == 'jump_empty':
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
            out_score_decimal=0
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
            elif rsec[0] > 0:
                y2, y1 = osec[1], osec[0]
                x2, x1 = rsec[1], rsec[0]
                a = (y2 - y1) / (x2 - x1)
                b = (y1 * x2 - y2 * x1) / (x2 - x1)
            else:
                a, b = 0, 0
            plt_formula.update({i: ((a, b),
                                    rsec,
                                    osec,
                                    'y = {:.8f}*x + {:.8f}'.format(a, b),
                                    )
                                })
            i += 1

        # function of formula
        def formula(x):
            for k in plt_formula.keys():
                # print(x, plt_formula[k])
                if (plt_formula[k][1][0] <= x <= plt_formula[k][1][1]) or \
                        (plt_formula[k][1][0] >= x >= plt_formula[k][1][1]):
                    return round45r(plt_formula[k][0][0] * x + plt_formula[k][0][1],
                                    out_score_decimal)
            return -1

        Result = namedtuple('Result', ('formula', 'coeff_raw_out_section_formula'))
        return Result(formula, plt_formula)

    @classmethod
    @run_timer
    def get_ppt_formula(cls,
                        raw_score_points,
                        raw_score_percent,
                        out_score_points,
                        out_score_ratio_cumu,
                        mode_ratio_prox='upper_min',
                        mode_ratio_cumu='no',
                        mode_sort_order='d',
                        mode_raw_score_max='map_by_ratio',
                        mode_raw_score_min='map_by_ratio',
                        out_score_decimal=0,
                        tiny_value=10**-12
                        ):
        ppt_formula = dict()
        _rmax, _rmin = max(raw_score_points), min(raw_score_points)
        if mode_sort_order in ['d', 'descending']:
            if any([x <= y for x, y in zip(raw_score_points[:-1], raw_score_points[1:])]):
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
            result = ModelAlgorithm.get_score_from_score_ratio_sequence(
                dest_ratio,
                out_score_points,
                out_score_ratio_cumu,
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
            ppt_formula.update({rscore: _seg})

        # function of formula
        def formula(x):
            if x in ppt_formula:
                return round45r(ppt_formula[x], out_score_decimal)
            else:
                return -1

        Result = namedtuple('Result', ('formula', 'map_dict'))
        return Result(formula, ppt_formula)

    @classmethod
    def get_tai_formula(cls,
                        df,
                        col,
                        pecent_first=0.01,
                        mode_ratio_prox='upper_min',
                        raw_score_max=100,
                        raw_score_min=0,
                        grade_num=15
                        ):

        map_table = run_seg(df=df,
                            cols=[col],
                            segmax=raw_score_max,
                            segmin=raw_score_min,
                            )

        section_list = [df[col].max()]
        for j in range(15):
            r = ModelAlgorithm.get_score_from_score_ratio_sequence(
                dest_ratio=0.01,
                seg_seq=map_table.seg,
                ratio_seq=map_table[col+'_percent']
            )

        def formula(x):
            return x

        Result = namedtuple('Result', ('formula', 'map_dict'))
        return Result(formula, map_table)

    @classmethod
    @run_timer
    def get_stm_score(cls,
                      df,
                      cols,
                      model_ratio_pdf,
                      model_section,
                      model_type='plt',
                      raw_score_max=100,
                      raw_score_min=0,
                      raw_score_step=1,
                      mode_ratio_cumu='no',
                      mode_ratio_prox='upper_min',
                      mode_sort_order='d',
                      mode_section_point_first='real',
                      mode_section_point_start='step',
                      mode_section_point_last='real',
                      mode_section_degraded='map_to_max',
                      mode_section_lost='ignore',
                      out_score_decimal=0,
                      ):
        seg = run_seg(df=df,
                      cols=cols,
                      segmax=raw_score_max,
                      segmin=raw_score_min,
                      segsort=mode_sort_order,
                      segstep=raw_score_step,
                      )
        map_table = seg.outdf
        cumu_ratio = [sum(model_ratio_pdf[0:i+1])/100 for i in range(len(model_ratio_pdf))]
        for col in cols:
            print('transform {} of {}'.format(col, cols))
            if model_type.lower() == 'plt':
                raw_section = ModelAlgorithm.get_raw_section(
                    section_ratio_cumu_sequence=cumu_ratio,
                    raw_score_sequence=map_table.seg,
                    raw_score_percent_sequence=map_table[col + '_percent'],
                    mode_ratio_cumu=mode_ratio_cumu,
                    mode_ratio_prox=mode_ratio_prox,
                    mode_sort_order=mode_sort_order,
                    mode_section_point_first=mode_section_point_first,
                    mode_section_point_start=mode_section_point_start,
                    mode_section_point_last=mode_section_point_last,
                    mode_section_lost=mode_section_lost,
                    raw_score_max=raw_score_max,
                    raw_score_min=raw_score_min,
                )
                formula = ModelAlgorithm.get_plt_formula(
                    raw_section=raw_section.section,
                    out_section=model_section,
                    mode_section_degraded=mode_section_degraded,
                    out_score_decimal=out_score_decimal
                )
            elif model_type.lower() == 'ppt':
                formula = ModelAlgorithm.get_ppt_formula(
                    raw_score_points=map_table.seg,
                    raw_score_percent=map_table[col+'_percent'],
                    out_score_points=[x[0] for x in model_section],
                    out_score_ratio_cumu=cumu_ratio,
                    mode_ratio_prox=mode_ratio_prox,
                    mode_ratio_cumu=mode_ratio_cumu,
                    mode_sort_order=mode_sort_order,
                    mode_raw_score_max=mode_section_point_first,
                    mode_raw_score_min=mode_section_point_last,
                    out_score_decimal=out_score_decimal
                )
            else:
                raise ValueError
            map_table.loc[:, col+'_ts'] = map_table.seg.apply(formula.formula)
            df[col+'_ts'] = df[col].apply(formula.formula)
        result=namedtuple('r', ['df', 'map_table'])
        return result(df, map_table)


# call SegTable.run() return instance of SegTable
def run_seg(
            df: pd.DataFrame,
            cols: list,
            segmax=100,
            segmin=0,
            segsort='d',
            segstep=1,
            display=False,
            usealldata=False
            ):
    seg = SegTable()
    seg.set_data(
        df=df,
        cols=cols
    )
    seg.set_para(
        segmax=segmax,
        segmin=segmin,
        segstep=segstep,
        segsort=segsort,
        display=display,
        useseglist=usealldata
    )
    seg.run()
    return seg


class SegTable(object):
    """
    * 分数分段及百分位表模型
    * model for score segment-percentile table
    * from 09-17-2017
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    # version 1.0.1 2018-09-24

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(df:DataFrame, fs:list)
        df: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        fs: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离，分段开始值，分数顺序，指定分段值列表， 使用指定分段列表，使用所有数据， 关闭计算过程显示信息
    set_para（segmax, segmin, segstep, segstart, segsort, seglist, useseglist, usealldata, display）
        segmax: int, maxvalue for segment, default=150
                输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。
                输出分段表中分数段的最小值
        segstep: int, grades for segment value, default=1
                分段间隔，用于生成n-分段表（五分一段的分段表）
        segstart:int, start seg score to count
                进行分段计算的起始值
        segsort: str, 'a' for ascending order or 'd' for descending order, default='d' (seg order on descending)
                输出结果中分段值得排序方式，d: 从大到小， a：从小到大
                排序模式的设置影响累计数和百分比的意义。
        seglist: list, used to create set value
                 使用给定的列表产生分段表，列表中为分段点值
        useseglist: bool, use or not use seglist to create seg value
                 是否使用给定列表产生分段值
        usealldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        display: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息
    outdf: 输出分段数据
            seg: seg value
        [field]: field name in fs
        [field]_count: number at the seg
        [field]_sum: cumsum number at the seg
        [field]_percent: percentage at the seg
        [field]_count[step]: count field for step != 1
        [field]_list: count field for assigned seglist when use seglist
    运行，产生输出数据, calculate and create output data
    run()

    应用举例
    example:
        import pyex_seg as sg
        seg = SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_para(segmax=100, segmin=1, segstep=1, segsort='d', usealldata=True, display=True)
        seg.run()
        print(seg.outdf.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          fs type is digit, for example: int or float

        3)可以单独设置数据(df),字段列表（fs),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.col = ['score_1', 'score_2'];
              seg.segmax = 120
          重新设置后需要运行才能更新输出数据ouput_data, 即调用run()
          便于在计算期间调整模型。
          by usting property mode, rawdata, scorefields, para can be setted individually
        4) 当设置大于1分的分段分值X时， 会在结果DataFrame中生成一个字段[segfiled]_countX，改字段中不需要计算的分段
          值设为-1。
          when segstep > 1, will create field [segfield]_countX, X=str(segstep), no used value set to -1 in this field
    """

    def __init__(self):
        # raw data
        self.__dfframe = None
        self.__cols = []

        # parameter for model
        self.__segList = []
        self.__useseglist = False
        self.__segStart = 100
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'd'
        self.__usealldata = True
        self.__display = True
        self.__percent_decimal = 10

        # result data
        self.__outdfframe = None

        # run status
        self.__run_completed = False

    @property
    def outdf(self):
        return self.__outdfframe

    @property
    def df(self):
        return self.__dfframe

    @df.setter
    def df(self, df):
        self.__dfframe = df

    @property
    def cols(self):
        return self.__cols

    @cols.setter
    def cols(self, cols):
        self.__cols = cols

    @property
    def seglist(self):
        return self.__segList

    @seglist.setter
    def seglist(self, seglist):
        self.__segList = seglist

    @property
    def useseglist(self):
        return self.__useseglist

    @useseglist.setter
    def useseglist(self, useseglist):
        self.__useseglist = useseglist

    @property
    def segstart(self):
        return self.__segStart

    @segstart.setter
    def segstart(self, segstart):
        self.__segStart = segstart

    @property
    def segmax(self):
        return self.__segMax

    @segmax.setter
    def segmax(self, segvalue):
        self.__segMax = segvalue

    @property
    def segmin(self):
        return self.__segMin

    @segmin.setter
    def segmin(self, segvalue):
        self.__segMin = segvalue

    @property
    def segsort(self):
        return self.__segSort

    @segsort.setter
    def segsort(self, sort_mode):
        self.__segSort = sort_mode

    @property
    def segstep(self):
        return self.__segStep

    @segstep.setter
    def segstep(self, segstep):
        self.__segStep = segstep

    @property
    def segalldata(self):
        return self.__usealldata

    @segalldata.setter
    def segalldata(self, datamode):
        self.__usealldata = datamode

    @property
    def display(self):
        return self.__display

    @display.setter
    def display(self, display):
        self.__display = display

    def set_data(self, df, cols=None):
        self.__dfframe = df
        if type(cols) == str:
            cols = [cols]
        if (not isinstance(cols, list)) & isinstance(df, pd.DataFrame):
            self.__cols = df.columns.values
        else:
            self.__cols = cols
        self.__check()

    def set_para(
            self,
            segmax=None,
            segmin=None,
            segstart=None,
            segstep=None,
            seglist=None,
            segsort=None,
            useseglist=None,
            usealldata=None,
            display=None):
        set_str = ''
        if segmax is not None:
            self.__segMax = segmax
            set_str += 'set segmax to {}'.format(segmax) + '\n'
        if segmin is not None:
            self.__segMin = segmin
            set_str += 'set segmin to {}'.format(segmin) + '\n'
        if segstep is not None:
            self.__segStep = segstep
            set_str += 'set segstep to {}'.format(segstep) + '\n'
        if segstart is not None:
            set_str += 'set segstart to {}'.format(segstart) + '\n'
            self.__segStart = segstart
        if isinstance(segsort, str):
            if segsort.lower() in ['d', 'a', 'D', 'A']:
                set_str += 'set segsort to {}'.format(segsort) + '\n'
                self.__segSort = segsort
        if isinstance(usealldata, bool):
            set_str += 'set segalldata to {}'.format(usealldata) + '\n'
            self.__usealldata = usealldata
        if isinstance(display, bool):
            set_str += 'set display to {}'.format(display) + '\n'
            self.__display = display
        if isinstance(seglist, list):
            set_str += 'set seglist to {}'.format(seglist) + '\n'
            self.__segList = seglist
        if isinstance(useseglist, bool):
            set_str += 'set seglistuse to {}'.format(useseglist) + '\n'
            self.__useseglist = useseglist
        if display:
            print(set_str)
        self.__check()
        if display:
            self.show_para()

    def show_para(self):
        print('------ seg para ------')
        print('    use seglist:{}'.format(self.__useseglist))
        print('        seglist:{}'.format(self.__segList))
        print('       maxvalue:{}'.format(self.__segMax))
        print('       minvalue:{}'.format(self.__segMin))
        print('       segstart:{}'.format(self.__segStart))
        print('        segstep:{}'.format(self.__segStep))
        print('        segsort:{}'.format('d (descending)' if self.__segSort in ['d', 'D'] else 'a (ascending)'))
        print('     usealldata:{}'.format(self.__usealldata))
        print('        display:{}'.format(self.__display))
        print('-' * 28)

    def help_doc(self):
        print(self.__doc__)

    def __check(self):
        if isinstance(self.__dfframe, pd.Series):
            self.__dfframe = pd.DataFrame(self.__dfframe)
        if not isinstance(self.__dfframe, pd.DataFrame):
            print('error: raw score data is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('error: segmax({}) is not greater than segmin({})!'.format(self.__segMax, self.__segMin))
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('error: segstep({}) is too small or big!'.format(self.__segStep))
            return False
        if not isinstance(self.__cols, list):
            if isinstance(self.__cols, str):
                self.__cols = [self.__cols]
            else:
                print('error: segfields type({}) error.'.format(type(self.__cols)))
                return False

        for f in self.__cols:
            if f not in self.df.columns:
                print("error: field('{}') is not in df fields({})".
                      format(f, self.df.columns.values))
                return False
        if not isinstance(self.__usealldata, bool):
            print('error: segalldata({}) is not bool type!'.format(self.__usealldata))
            return False
        return True

    def run(self):
        sttime = time.clock()
        if not self.__check():
            return
        # create output dataframe with segstep = 1
        if self.__display:
            print('---seg calculation start---')
        seglist = [x for x in range(int(self.__segMin), int(self.__segMax + 1))]
        if self.__segSort in ['d', 'D']:
            seglist = sorted(seglist, reverse=True)
        self.__outdfframe = pd.DataFrame({'seg': seglist})
        outdf = self.__outdfframe
        for f in self.__cols:
            # calculate preliminary group count
            tempdf = self.df
            tempdf.loc[:, f] = tempdf[f].apply(round45r)

            # count seg_count in [segmin, segmax]
            r = tempdf.groupby(f)[f].count()
            # fcount_list = [np.int64(r[x]) if x in r.index else 0 for x in seglist]
            outdf.loc[:, f+'_count'] = [np.int64(r[x]) if x in r.index else 0 for x in seglist]
            if self.__display:
                print('finished count(' + f, ') use time:{}'.format(time.clock() - sttime))

            # add outside scope number to segmin, segmax
            if self.__usealldata:
                outdf.loc[outdf.seg == self.__segMin, f + '_count'] = \
                    r[r.index <= self.__segMin].sum()
                outdf.loc[outdf.seg == self.__segMax, f + '_count'] = \
                    r[r.index >= self.__segMax].sum()

            # calculate cumsum field
            outdf[f + '_sum'] = outdf[f + '_count'].cumsum()
            if self.__useseglist:
                outdf[f + '_list_sum'] = outdf[f + '_count'].cumsum()

            # calculate percent field
            maxsum = max(max(outdf[f + '_sum']), 1)     # avoid divided by 0 in percent computing
            outdf[f + '_percent'] = \
                outdf[f + '_sum'].apply(lambda x: round45r(x/maxsum, self.__percent_decimal))
            if self.__display:
                print('segments count finished[' + f, '], used time:{}'.format(time.clock() - sttime))

            # self.__outdfframe = outdf.copy()
            # special seg step
            if self.__segStep > 1:
                self.__run_special_step(f)

            # use seglist
            if self.__useseglist:
                if len(self.__segList) > 0:
                    self.__run_seg_list(f)

        if self.__display:
            print('segments count total consumed time:{}'.format(time.clock()-sttime))
            print('---seg calculation end---')
        self.__run_completed = True
        self.__outdfframe = outdf
        return

    def __run_special_step(self, field: str):
        """
        processing count for step > 1
        :param field: for seg stepx
        :return: field_countx in outdf
        """
        f = field
        segcountname = f + '_count{0}'.format(self.__segStep)
        self.__outdfframe[segcountname] = np.int64(-1)
        curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        curpoint = self.__segStart
        if self.__segSort.lower() == 'd':
            while curpoint+curstep > self.__segMax:
                curpoint += curstep
        else:
            while curpoint+curstep < self.__segMin:
                curpoint += curstep
        cum = 0
        for index, row in self.__outdfframe.iterrows():
            cum += row[f + '_count']
            curseg = np.int64(row['seg'])
            if curseg in [self.__segMax, self.__segMin]:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if (self.__segStart <= self.__segMin) | (self.__segStart >= self.__segMax):
                    curpoint += curstep
                continue
            if curseg in [self.__segStart, curpoint]:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                curpoint += curstep

    def __run_seg_list(self, field):
        """
        use special step list to create seg
        calculating based on field_count
        :param field:
        :return:
        """
        f = field
        segcountname = f + '_list'
        self.__outdfframe[segcountname] = np.int64(-1)
        segpoint = sorted(self.__segList) \
            if self.__segSort.lower() == 'a' \
            else sorted(self.__segList)[::-1]
        # curstep = self.__segStep if self.__segSort.lower() == 'a' else -self.__segStep
        # curpoint = self.__segStart
        cum = 0
        pos = 0
        curpoint = segpoint[pos]
        rownum = len(self.__outdfframe)
        cur_row = 0
        lastindex = 0
        maxpoint = max(self.__segList)
        minpoint = min(self.__segList)
        list_sum = 0
        self.__outdfframe.loc[:, f+'_list_sum'] = 0
        for index, row in self.__outdfframe.iterrows():
            curseg = np.int64(row['seg'])
            # cumsum
            if self.__usealldata | (minpoint <= curseg <= maxpoint):
                cum += row[f + '_count']
                list_sum += row[f+'_count']
                self.__outdfframe.loc[index, f+'_list_sum'] = np.int64(list_sum)
            # set to seg count, only set seg in seglist
            if curseg == curpoint:
                self.__outdfframe.loc[index, segcountname] = np.int64(cum)
                cum = 0
                if pos < len(segpoint)-1:
                    pos += 1
                    curpoint = segpoint[pos]
                else:
                    lastindex = index
            elif cur_row == rownum:
                if self.__usealldata:
                    self.__outdfframe.loc[lastindex, segcountname] += np.int64(cum)
            cur_row += 1
# SegTable class end


# deprecated round45 function
def round45r_old2(number, digits=0):
    """
    float is not precise at digit 16 from decimal point.
    if hope that round(1.265, 3): 1.264999... to 1.265000...
    need to add a tiny error to 1.265: round(1.265 + x*10**-16, 3) => 1.265000...
    note that:
        10**-16     => 0.0...00(53)1110011010...
        2*10**-16   => 0.0...0(52)1110011010...
        1.2*10**-16 => 0.0...0(52)100010100...
    so 10**-16 can not definitely represented in float 1+52bit

    (16 - int_len) is ok, 17 is unstable
    test result:
    format(1.18999999999999994671+10**-16, '.20f')     => '1.1899999999999999(16)4671'      ## digit-16 is reliable
    format(1.18999999999999994671+2*10**-16, '.20f')   => '1.1900000000000001(16)6875'
    format(1.18999999999999994671+1.2*10**-16, '.20f') => '1.1900000000000001(16)6875'
    format(1.18999999999999994671+1.1*10**-16, '.20f') => '1.1899999999999999(16)4671'
    """

    int_len = str(abs(number)).find('.')
    if int_len + digits > 16:
        print('float cannot support {} digits precision'.format(digits))
        raise ValueError
    add_err = 10**-12       # valid for 0-16000
    # add_err = 3.55275*10**-15
    # add_err = 2*10**-14
    # add_err = 2 * 10 ** -(16 - int_len + 1) * (1 if number > 0 else -1)
    # if format(number, '.' + str(16 - digits - int_len) + 'f').rstrip('0') <= str(number):
    #     return round(number + add_err, digits) + add_err
    return round(number+add_err, digits)


# deprecated
def round45r_old1(number, digits=0):
    __doc__ = '''
    use multiple 10 power and int method
    precision is not normal at decimal >16 because of binary representation
    :param number: input float value
    :param digits: places after decimal point
    :return: rounded number with assigned precision
    '''
    if format(number, '.'+str(digits+2)+'f').rstrip('0') <= str(number):
        return round(number+10**-(digits+2), digits)
    return round(number, digits)
