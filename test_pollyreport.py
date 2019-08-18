# -*- utf8 -*-

import PollyReports as prt
import reportlab.pdfgen.canvas as canvas
import pyex_bzy as bzy


def test(f='d:/test.pdf', df=None):
    cv1 = canvas.Canvas(f)
    data_names = df.columns
    datasource = [{data_names[j]: row[1][j] for j in range(len(data_names))}
                  for row in df.iterrows()]
    rp1 = prt.Report(datasource=datasource)
    rp1.detailband = prt.Band([
        prt.Element((10+j*45, 0), ("Helvetica", 12), key=name, align='left')
        for j, name in enumerate(data_names)])
    rp1.pageheader = prt.Band([
        prt.Element((10+j*45, 0), ("Helvetica", 12), text=name, align='left')
                                  for j,name in enumerate(data_names)] +
        [prt.Rule((10, 18), 7.5 * 65, thickness=0.5)])
    rp1.pagefooter = prt.Band([prt.Rule((10, 0), 7.5 * 65, thickness=1)])

    # rp1.pageheader = 'High Test 2019 Segment Table'
    rp1.generate(cv1)
    cv1.save()
