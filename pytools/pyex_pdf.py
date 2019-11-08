# -*- utf8 -*-

from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open

fs = 'f:/studies/lqdata/fd2018ck.pdf'

def readPDF(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content

def parsepdf(fs=fs):
    # pdfFile = urlopen("http://pythonscraping.com/pages/warandpeace/chapter1.pdf")
    pdfFile = open(fs)
    outputString = readPDF(pdfFile)
    print(outputString)
    pdfFile.close()
