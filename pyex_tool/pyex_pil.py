# -*- utf8 -*-

from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
import os
import pandas as pd


# exif info from PIL
doc1 = """
        ExifVersion
        ComponentsConfiguration
        ExifImageWidth
        DateTimeOriginal
        DateTimeDigitized
        ExifInteroperabilityOffset
        FlashPixVersion
        MeteringMode
        LightSource
        Flash
        FocalLength
        41986
        ImageDescription
        Make
        Model
        Orientation
        YCbCrPositioning
        41988
        XResolution
        YResolution
        59932
        ExposureTime
        ExposureProgram
        ColorSpace
        41990
        ISOSpeedRatings
        ResolutionUnit
        41987
        FNumber
        Software
        DateTime
        ExifImageHeight
        ExifOffset
        """
# exif info from modul exif
doc2 = """
        EXIF ColorSpace (Short): sRGB
        EXIF ComponentsConfiguration (Undefined): YCbCr
        EXIF DateTimeDigitized (ASCII): 2012:11:22 15:35:14
        EXIF DateTimeOriginal (ASCII): 2012:11:22 15:35:14
        EXIF DigitalZoomRatio (Ratio): 1
        EXIF ExifImageLength (Long): 2560
        EXIF ExifImageWidth (Long): 1920
        EXIF ExifVersion (Undefined): 0220
        EXIF ExposureBiasValue (Signed Ratio): 0
        EXIF ExposureMode (Short): Auto Exposure
        EXIF ExposureProgram (Short): Portrait Mode
        EXIF ExposureTime (Ratio): 1/256
        EXIF FNumber (Ratio): 14/5
        EXIF Flash (Short): Flash did not fire
        EXIF FlashPixVersion (Undefined): 0100
        EXIF FocalLength (Ratio): 35
        EXIF ISOSpeedRatings (Short): 56
        EXIF InteroperabilityOffset (Long): 4810
        EXIF LightSource (Short): other light source
        EXIF MeteringMode (Short): CenterWeightedAverage
        EXIF Padding (Undefined): []
        EXIF SceneCaptureType (Short): Portrait
        EXIF WhiteBalance (Short): Auto
        Image DateTime (ASCII): 2012:11:24 09:44:50
        Image ExifOffset (Long): 2396
        Image ImageDescription (ASCII):
        Image Make (ASCII):
        Image Model (ASCII):
        Image Orientation (Short): Horizontal (normal)
        Image Padding (Undefined): []
        Image ResolutionUnit (Short): Pixels/Inch
        Image Software (ASCII): Microsoft Windows Photo Viewer 6.1.7600.16385
        Image XResolution (Ratio): 72
        Image YCbCrPositioning (Short): Co-sited
        Image YResolution (Ratio): 72
        Thumbnail Compression (Short): JPEG (old-style)
        Thumbnail JPEGInterchangeFormat (Long): 4970
        Thumbnail JPEGInterchangeFormatLength (Long): 3883
        Thumbnail Orientation (Short): Horizontal (normal)
        Thumbnail ResolutionUnit (Short): Pixels/Inch
        Thumbnail XResolution (Ratio): 72
        Thumbnail YCbCrPositioning (Short): Co-sited
        Thumbnail YResolution (Ratio): 72
        """


class Exif(object):
    def __init__(self, file_name=None):
        self.image = None
        self.file_name = file_name
        self.file_path = None
        if isinstance(file_name, str):
            self.load_image(self.file_name)

    def load_image(self, file_name=None):
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str):
            print('file name is empty!')
            return False
        self.file_name = file_name
        p = file_name.rfind('/') if file_name.rfind('/') else file_name.rfind('\\')
        self.file_path = file_name[:p] if p > 0 else ''
        if os.path.isfile(file_name):
            try:
                self.image = Image.open(file_name)
                self.__get_exif()
                # self.image_data = self.image
                # self.image.close()
            except IOError:
                print('IOERROR ' + file_name)
                return False
        elif os.path.isdir(self.file_path):
            print('file path=({}) exists, \nfile name is error!'.format(self.file_path))
            return False
        else:
            print('file name error =({})!')
            return False
        return True

    def show_image(self):
        if self.image is not None:
            plt.imshow(self.image)

    def show_exif(self):
        if self.exif_info is not None:
            print(self.exif_info)

    def __get_exif(self, file_name=None, reload=False):
        _file_name = file_name
        if file_name is None:
            _file_name = self.file_name
            if (self.image is None) or reload:
                if not self.load_image():
                    return None
        else:
            if not self.load_image(file_name):
                return None
        get_exif = {'exif_items': [], 'exif_content': []}  # 'no exif'
        if hasattr(self.image, '_getexif'):
            exifinfo = self.image._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)
                    # get_exif[decoded] = value
                    get_exif['exif_items'].append(format(decoded, '30s'))
                    value = value if len(str(value)) < 50 else str(value)[:50]
                    get_exif['exif_content'].append(format(str(value), '50s'))
        self.exif_info = pd.DataFrame(get_exif)
        return self.exif_info
