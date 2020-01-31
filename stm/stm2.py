# coding: utf-8


from collections import namedtuple
import numpy as np
import scipy.stats as sts
# import pandas as pd

from stm import stmlib as slib


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
        get pdf, cdf, cutoff_err for defined model: start, end, section_num, std_num, model_type
        add tail_ratio(cutoff_err) to first and end section from norm table
        note: std_num = std number from mean to end point

        example:
           get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.9508).pdf
           [0.03000, 0.07513, 0.16036, 0.234265, ..., 0.03000],
           get_section_pdf(start=21, end=100, section_num=8, std_num = 40/15.6065).pdf
           [0.02729, 0.07272, 0.16083, 0.23916, ..., 0.02729],
           it means that std==15.95 is fitting ratio 0.03,0.07 in the table

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
            pdf_table[0] += cutoff
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
        # if dest_ratio > 1+tiny_value:
        #     return Result(False, False, True,
        #                   -200, -200,
        #                   1, 1,
        #                   999, 999
        #                   )

        last_percent = -1
        last_seg = -1
        _top, _bottom, _len = False, False, len(seg_seq)
        for row_id, (seg, percent) in enumerate(zip(seg_seq, ratio_seq)):
            this_percent = percent
            this_seg = seg
            if row_id == (_len-1):
                _bottom = True
                this_seg_near = True
                _top = False
                dist_to_this = 900
                dist_to_last = 999
                return Result(this_seg_near, _top, _bottom,
                              this_seg, last_seg,
                              float(this_percent), float(last_percent),
                              float(dist_to_this), float(dist_to_last),
                              )
            # meet a percent that bigger or at table bottom
            if this_percent >= dest_ratio:
                if row_id == 0:
                    _top = True
                dist_to_this = abs(float(this_percent - dest_ratio))
                dist_to_last = abs(float(dest_ratio - last_percent))
                if _top:    # at top and percent >= ratio
                    dist_to_this = abs(float(this_percent - dest_ratio))
                    dist_to_last = 999
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
        return Result(False, False, False, -11, -11, -1, -1, 999, 999)

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
            mode_section_lost='real',
            raw_score_max=100,
            raw_score_min=0,
            tiny_value=10**-8,
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
        dest_ratio_list = []

        # step-2: get end-point of each section
        section_percent_list = []
        dest_ratio = None
        last_ratio = 0
        real_percent = 0
        _bottom = False
        for ratio in section_ratio_cumu_sequence:
            if dest_ratio is None:
                dest_ratio = ratio
            else:
                if mode_ratio_cumu == 'yes':
                    dest_ratio = real_percent + ratio - last_ratio
                else:
                    dest_ratio = ratio
            dest_ratio_list.append(dest_ratio)

            # avoid to locate invalid ratio
            _seg, _percent = -1, 1
            if not _bottom:
                result = ModelAlgorithm.get_score_from_score_ratio_sequence(
                    dest_ratio,
                    raw_score_sequence,
                    raw_score_percent_sequence,
                    tiny_value)
                # print(result)
                # at bottom, last search for sequence
                #   set bottom value and avoid to search again
                if result.bottom:
                    _seg = [x for x in raw_score_sequence][-1]
                    _percent = 1
                    _bottom = True  # avoid to repeat search
                # at top, single point for first section
                elif result.top:
                    _seg, _percent = result.this_seg, result.this_percent
                # strategy: mode_ratio_prox:
                # equal to this, prori to mode
                elif result.dist_to_this < tiny_value:
                    _seg, _percent = result.this_seg, result.this_percent
                # equal to last, prior to mode
                elif result.dist_to_last < tiny_value:
                    _seg, _percent = result.last_seg, result.last_percent
                elif mode_ratio_prox == 'lower_max':
                    _seg, _percent = result.last_seg, result.last_percent
                elif mode_ratio_prox == 'upper_min':
                    _seg, _percent = result.this_seg, result.this_percent
                # near or 'near_max' or 'near_min'
                elif 'near' in mode_ratio_prox:
                    # same dist
                    if abs(result.dist_to_this - result.dist_to_last) < tiny_value:
                        if mode_ratio_prox == 'near_max':
                            _seg, _percent = result.this_seg, result.this_percent
                        else:
                            _seg, _percent = result.last_seg, result.last_percent
                    # near to this
                    elif result.dist_to_this < result.dist_to_last:
                        _seg, _percent = result.this_seg, result.this_percent
                    else:
                        _seg, _percent = result.last_seg, result.last_percent
                else:
                    # print('mode_ratio_prox error: {}'.format(mode_ratio_prox))
                    raise ValueError
            section_point_list.append(_seg)
            section_percent_list.append(_percent)
            last_ratio = ratio
            if _percent > 0:    # jump over lost section
                real_percent = _percent

        # print(section_point_list)
        #step-3-3: process last point
        if mode_section_point_last == 'defined':
            _last_value = raw_score_min if mode_sort_order in ['d', 'descending'] else \
                          raw_score_max
            i = 0
            while i < len(section_point_list)-1:
                if section_point_list[-1-i] >= 0:
                    if section_point_list[-1-i] != _last_value:
                        section_point_list[-1-i] = _last_value
                    break
                i += 1

        # step-4: make section
        #   with strategy: mode_section_point_start
        #                  default: step
        section_list = []
        lost = False
        for i, (x, y) in enumerate(zip(section_point_list[0:-1], section_point_list[1:])):
            _x, _y = None, None
            _step = 1 if mode_sort_order in ['a', 'ascending'] else -1
            if x == y:
                if i > 0:
                    # middle or last lost
                    _x, _y = -1, -1
                    lost = True
                else:
                    # first section
                    _x, _y = x, y
            else:
                if mode_section_point_start == 'share':
                    if i == 0:
                        # first section
                        _x, _y = x, y
                    else:
                        if (x > 0) and (y >= 0):
                            if lost:
                                # last lost, no ajacent section !
                                # must to step
                                _x, _y = x + _step, y
                                lost = False    # resume section state
                            else:
                                _x, _y = x, y
                        else:
                            _x, _y = -1, -1
                            lost = True
                else:
                    if i == 0:
                        # first section
                        _x, _y = x, y
                    else:
                        if (x >= 0) and (y >= 0):
                            forward_ok = (x + _step >= y) \
                                         if mode_sort_order in ['d', 'desceding'] \
                                         else (x + _step <= y)
                            if not forward_ok:
                                _x, _y = -1, -1
                                lost = True
                            else:
                                _x, _y = x + _step, y
                        else:
                            _x, _y = -1, -1
            if (mode_section_lost == 'zip') and (_x + _y == -2):
                continue
            section_list.append((_x, _y))

        # step-5: add lost section with (-1, -1)
        less_len = len(section_ratio_cumu_sequence) - len(section_list)
        if less_len > 0:
            section_list += [(-1, -1)] * less_len
            section_percent_list += [1] * less_len

        section_list = [x if ((x[0] >= 0) and (x[1] >= 0)) else (-1, -1)
                        for x in section_list]

        Result = namedtuple('result', ['section', 'point', 'dest_ratio', 'real_ratio'])
        return Result(section_list, section_point_list, dest_ratio_list, section_percent_list)

    @classmethod
    def get_plt_formula(cls,
            raw_section,
            out_section,
            mode_section_degraded='to_max',
            mode_sort_order='d',
            out_score_decimal=0
            ):
        plt_formula = dict()
        i = 0
        for rsec, osec in zip(raw_section, out_section):
            # rsec is degraded
            if rsec[0] == rsec[1]:
                a = 0
                if mode_section_degraded == 'to_max':
                    b = max(osec)
                elif mode_section_degraded == 'to_min':
                    b = min(osec)
                elif mode_section_degraded == 'to_mean':
                    b = np.mean(osec)
                else:
                    raise ValueError
                y0 = b
                x0 = osec[0]
            elif abs(rsec[0]-rsec[1]) > 0:
                if mode_sort_order in ['d', 'descending']:
                    y2, y1 = osec[0], osec[1]
                    x2, x1 = rsec[0], rsec[1]
                    a = (y2 - y1) / (x2 - x1)
                    b = (y1 * x2 - y2 * x1) / (x2 - x1)
                    y0 = osec[1]
                else:
                    y2, y1 = osec[1], osec[0]
                    x2, x1 = rsec[1], rsec[0]
                    a = (y2 - y1) / (x2 - x1)
                    b = (y1 * x2 - y2 * x1) / (x2 - x1)
                    y0 = osec[0]
                x0 = x1
            else:
                a, b, y0, x0 = 0, 0, 0, 0
            a, b = (a, b) if rsec[0] >=0 else (0, 0)
            formula_str1 = '***' if rsec[0] < 0 else 'y = {:.6f} * x {:+10.6f}'.format(a, b)
            formula_str2 = '***' if rsec[0] < 0 else \
                           'y = {0:8.6f} * (x - {1:10.6f}) + {2:10.6f}'.format(a, x0, y0)
            plt_formula.update({i+1: ((a, b), rsec, osec, formula_str1, formula_str2)})
            i += 1

        # function of formula
        def formula(x):
            for k in plt_formula.keys():
                if (plt_formula[k][1][0] <= x <= plt_formula[k][1][1]) or \
                        (plt_formula[k][1][0] >= x >= plt_formula[k][1][1]):
                    return slib.round45r(plt_formula[k][0][0] * x + plt_formula[k][0][1],
                                         out_score_decimal)
            return -1000

        Result = namedtuple('Result', ('formula', 'formula_dict'))
        return Result(formula, plt_formula)

    @classmethod
    # @slib.timer_wrapper
    def get_ppt_formula(cls,
                        raw_score_points,
                        raw_score_percent,
                        out_score_points,
                        out_score_ratio_cumu,
                        mode_ratio_prox='upper_min',
                        mode_ratio_cumu='no',
                        mode_sort_order='d',
                        mode_raw_score_max='real',  # map by real ratio
                        mode_raw_score_min='real',  # map by real ratio
                        out_score_decimal=0,
                        tiny_value=10**-12
                        ):
        map_score = dict()
        _rmax, _rmin = max(raw_score_points), min(raw_score_points)
        if mode_sort_order in ['d', 'descending']:
            if any([x <= y for x, y in zip(raw_score_points[:-1], raw_score_points[1:])]):
                print('raw score sequence is not correct order: {}'.format(mode_sort_order))
                return

        # lcoate out-score to raw-ratio in out-score-ratio-sequence
        dest_ratio_list = []
        real_ratio_list = []
        dest_ratio = None
        last_ratio = 0
        real_percent = 0
        for rscore, raw_ratio in zip(raw_score_points, raw_score_percent):
            if dest_ratio is None:
                dest_ratio = raw_ratio
            else:
                if mode_ratio_cumu == 'yes':
                    dest_ratio = real_percent + raw_ratio - last_ratio
                else:
                    dest_ratio = raw_ratio
            dest_ratio_list.append(dest_ratio)
            if rscore == _rmax:
                if mode_raw_score_max == 'defined':
                    map_score.update({rscore: max(out_score_points)})
                    real_ratio_list.append(0)
                    continue
            if rscore == _rmin:
                if mode_raw_score_min == 'defined':
                    map_score.update({rscore: min(out_score_points)})
                    real_ratio_list.append(1)
                    continue

            # set invalid ratio if can not found ration in out_percent
            _seg, _percent = -1, -1
            result = ModelAlgorithm.get_score_from_score_ratio_sequence(
                dest_ratio,
                out_score_points,
                out_score_ratio_cumu,
                tiny_value)

            # strategy: mode_ratio_prox:
            # choose this_seg if near equal to this or upper_min
            if (result.dist_to_this < tiny_value) or (mode_ratio_prox == 'upper_min'):
                _seg, _percent = result.this_seg, result.this_percent
            # choose last_seg if last is near or equal
            elif (mode_ratio_prox == 'lower_max') or (result.dist_to_last < tiny_value):
                if not result.top:
                    _seg, _percent = result.last_seg, result.last_percent
                else:
                    _seg, _percent = result.this_seg, result.this_percent
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
            map_score.update({rscore: _seg})
            real_ratio_list.append(_percent)

        # function of formula
        def formula(x):
            if x in map_score:
                return slib.round45r(map_score[x], out_score_decimal)
            else:
                # set None to raise ValueError in score calculating to avoid create misunderstand
                return -2000

        # print(dest_ratio, '\n', real_ratio_list)
        Result = namedtuple('Result', ('formula', 'map_dict', 'dest_ratio', 'real_ratio', 'map_table'))
        return Result(formula, map_score, dest_ratio_list, real_ratio_list,
                      list(zip(out_score_points, out_score_ratio_cumu)))

    @classmethod
    def get_pgt_formula(cls,
                        df=None,
                        col=None,
                        map_table=None,
                        percent_first=0.01,
                        mode_ratio_prox='upper_min',
                        mode_sort_order='d',
                        raw_score_max=100,
                        raw_score_min=0,
                        grade_num=15
                        ):

        section_point_list = [df[col].max()]
        r = ModelAlgorithm.get_score_from_score_ratio_sequence(
            dest_ratio=percent_first,
            seg_seq=map_table.seg,
            ratio_seq=map_table[col+'_percent']
        )

        top_level_score = None
        if mode_ratio_prox == 'upper_min' or r.bottom or r.top:
            top_level_score=r.this_seg
        elif mode_ratio_prox == 'lower_max':
            top_level_score=r.last_seg
        elif 'near' in mode_ratio_prox:
            if r.this_seg_near:
                top_level_score=r.this_seg
            else:
                top_level_score=r.last_seg
        top_level_score = slib.round45r(df.query(col+'>='+str(top_level_score))[col].mean(), 0)
        section_point_list.append(top_level_score)

        _step = -1 if mode_sort_order in ['d', 'descending'] else 1

        # use float value for grade step
        # to avoid to increase much cumulative error in last section
        grade_step = (top_level_score - raw_score_min)/(grade_num-1)
        for j in range(grade_num-1):
            section_point_list.append(slib.round45r(top_level_score+grade_step*_step*(j+1), 0))
        section_list = []
        for i, (x, y) in enumerate(zip(section_point_list[:-1], section_point_list[1:])):
            if i == 0:
                section_list.append((x, y))
            else:
                section_list.append((x+_step, y))
        # print(section_point_list, '\n', section_list)

        map_dict = dict()
        for si, sp in enumerate(section_point_list[1:]):  # grade_level == si+1
            if si == 0:
                for ss in range(section_point_list[si], sp+_step, _step):
                    map_dict.update({ss: si+1})
            else:
                for ss in range(section_point_list[si]+_step, sp+_step, _step):
                    map_dict.update({ss: si+1})
        
        def formula(x):
            if x in map_dict.keys():
                return map_dict[x]
            else:
                return -3000

        Result = namedtuple('Result', ('formula', 'section', 'map_dict', 'grade_step', 'top_level'))
        return Result(formula, section_list, map_dict, grade_step, top_level_score)

    @classmethod
    # @slib.timer_wrapper
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
                      mode_section_degraded='to_max',
                      mode_section_lost='real',
                      mode_score_zero='real',
                      out_score_decimals=0,
                      tiny_value=10**-8,
                      logger=None,
                      ):
        # start stm
        if isinstance(cols, tuple):
            cols = list(cols)
        elif isinstance(cols, str):
            cols = [cols]

        if logger:
            logger.loginfo('stm2 start ...')

        # create seg_table
        seg = slib.run_seg(
              df=df,
              cols=cols,
              segmax=raw_score_max,
              segmin=raw_score_min,
              segsort=mode_sort_order,
              segstep=raw_score_step,
              display=False,
              )
        map_table = seg.outdf
        section_num = len(model_ratio_pdf)
        if mode_sort_order in ['d', 'descending']:
            cumu_ratio = [sum(model_ratio_pdf[0:i+1])/100 for i in range(section_num)]
        else:
            _ratio_list = model_ratio_pdf[::-1]
            cumu_ratio = [sum(_ratio_list[0:i + 1]) / 100 for i in range(section_num)]
            model_section = [tuple(reversed(x)) for x in reversed(model_section)]

        # start transform
        result = None
        for col in cols:
            if logger:
                logger.loginfo('transform {} of {}'.format(col, cols))
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
                    tiny_value=tiny_value,
                )
                result = ModelAlgorithm.get_plt_formula(
                    raw_section=raw_section.section,
                    out_section=model_section,
                    mode_section_degraded=mode_section_degraded,
                    mode_sort_order=mode_sort_order,
                    out_score_decimal=out_score_decimals
                    )
                formula = result.formula

                # display ratio searching result at each section
                if logger:
                    for i, (c_ratio, d_ratio, raw_sec, r_ratio, out_sec) \
                            in enumerate(zip(
                                            cumu_ratio,
                                            raw_section.dest_ratio,
                                            raw_section.section,
                                            raw_section.real_ratio,
                                            model_section
                                            )):
                        logger.loginfo(
                            '   <{0:02d}> ratio: [def:{1:.4f}  dest:{2:.4f}  match:{3:.4f}] => '
                            'section_map: raw:[{4:3d}, {5:3d}] --> out: [{6:3d}, {7:3d}]'.
                            format(i + 1,
                                  c_ratio,
                                  d_ratio,
                                  r_ratio,
                                  raw_sec[0],
                                  raw_sec[1],
                                  int(out_sec[0]),
                                  int(out_sec[1]),
                                  )
                              )
                    for k in result.formula_dict.keys():
                        logger.loginfo('   [{0:02d}]: {1}'.format(k, result.formula_dict[k][4]))
                    # logger.loginfo('='*100)
            elif model_type.lower() == 'ppt':
                result = ModelAlgorithm.get_ppt_formula(
                    raw_score_points=map_table.seg,
                    raw_score_percent=map_table[col+'_percent'],
                    out_score_points=[x[0] for x in model_section],
                    out_score_ratio_cumu=cumu_ratio,
                    mode_ratio_prox=mode_ratio_prox,
                    mode_ratio_cumu=mode_ratio_cumu,
                    mode_sort_order=mode_sort_order,
                    mode_raw_score_max=mode_section_point_first,
                    mode_raw_score_min=mode_section_point_last,
                    out_score_decimal=out_score_decimals
                    )
                formula = result.formula
                if logger:
                    logger.loginfo(
                        ' model table: [{0}]\n'
                        'real percent: [{1}]\n'
                        '   get ratio: [{2}]\n'
                        '   raw score: [{3}]\n'
                        '   out score: [{4}]\n'
                        .format(
                        ', '.join([format(int(x), '3d')+':'+format(y, '8.6f')
                                  for (x, z), y in zip(model_section, cumu_ratio)
                                  if x in result.map_dict.values()]),
                        ', '.join([format(x, '12.8f') for x in result.dest_ratio]),
                        ', '.join([format(x, '12.8f') for x in result.real_ratio]),
                        ', '.join([format(x, '>12d') for x in map_table.seg]),
                        ', '.join([format(slib.round45r(result.formula(x), out_score_decimals), '>' +
                                   ('12d' if out_score_decimals == 0 else '12.' + str(out_score_decimals) + 'f'))
                                   for x in map_table.seg]),
                        ))
            elif model_type.lower() == 'pgt':
                result = ModelAlgorithm.get_pgt_formula(
                    df=df,
                    col=col,
                    map_table=map_table,
                    percent_first=model_ratio_pdf[0]/100,
                    mode_ratio_prox=mode_ratio_prox,
                    mode_sort_order=mode_sort_order,
                    raw_score_max=raw_score_max,
                    raw_score_min=raw_score_min,
                    grade_num=len(model_section)
                    )
                if logger:
                    logger.loginfo('tai score section: {}'.format(result.section))
                    logger.loginfo('       grade step: {}'.format(result.grade_step))
                    logger.loginfo('        top level: {}'.format(result.top_level))
                formula = result.formula
            else:
                raise ValueError
            map_table.loc[:, col+'_ts'] = map_table.seg.apply(formula)
            df[col+'_ts'] = df[col].apply(formula)

            if out_score_decimals == 0:
                df = df.astype({col+'_ts': int})

        if logger:
            logger.loginfo('stm2 running end \n' + '-'*100)
        r = namedtuple('r', ['outdf', 'map_table', 'result'])
        return r(df, map_table, result)
