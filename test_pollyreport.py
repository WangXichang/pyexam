# -*- utf8 -*-

import PollyReports as prt
import reportlab.pdfgen.canvas as canvas
import pyex_bzy as bzy


def test(f='d:/test.pdf', df=None):
    cv1 = canvas.Canvas(f)
    datasource = [{'fd': row[1][0], 'wk': row[1][1], 'wklj': row[1][2]}
                  for row in df.iterrows()]
    rp1 = prt.Report(datasource=datasource)
    rp1.detailband = prt.Band([
        prt.Element((10, 0), ("Helvetica", 12), key='fd'),
        prt.Element((60, 0), ("Helvetica", 12), key='wk'),
        prt.Element((100, 0), ("Helvetica", 12), key='wklj')
    ])
    rp1.pageheader = prt.Band([
        prt.Element((10, 0), ("Helvetica", 12), text='fd'),
        prt.Element((60, 0), ("Helvetica", 12), text='wk'),
        prt.Element((100, 0), ("Helvetica", 12), text='wklj'),
        prt.Rule((10, 20), 7.5 * 60, thickness=2)
    ])
    # rp1.pageheader = 'High Test 2019 Segment Table'
    rp1.generate(cv1)
    cv1.save()
