#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Live widget next gen demonstration."""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QPrinter, QPrintDialog, QPainter
from PyQt4.uic import loadUi

import numpy

# Allow running this after "python setup.py build"
sys.path[0:0] = glob.glob('../pyfc/build/lib.*')
sys.path[0:0] = glob.glob('../pyfcfits/build/lib.*')
sys.path[0:0] = glob.glob('../pyipw/build/lib.*')

from nicos_filterchain import QtValidator, ImageOutputChain, Numpy2d, InvokeFilter, FixDimFilter2d, FilterVar
from nicos_filterchain_fits import Fits2d
from nicos_imageplot import ImagePlot

# from nicosfilterchain import __VERSION__ as nicosfilterchainversion
# from nicosimagewidget import __VERSION__ as nicosimagewidgetversion


class MainWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        ui = loadUi('demo.ui', self)

        validator = QtValidator(self)

        outputChain = ImageOutputChain()

        data1 = numpy.identity(3)
        data1 = numpy.array(data1, dtype='f')

        di = [[1,1,1],[1,1,1],[1,5,1]]

        ob = [[5,5,5],[9,9,9],[5,9,5]]

        im = [[5,3,4],[1,5,7],[5,7,4]]

        npfilter = Numpy2d()
        npfilter.setData(im)

        fitsfilter = Fits2d()
        fitsfilter.setFilename('C:/Dev/huge data/Tomography/lava/raw/seismolava64__000.000.fits')

        # filterChain.setInputFitsFile('/home/felix/Projekte/daten/lava/raw/seismolava64__000.000.fits')

        # filterChain.setInputFitsFile('C:/Dev/huge data/Tomography/lava/raw/seismolava64__000.000.fits')
        # filterChain.setDarkImageFitsFile('C:/Dev/huge data/Tomography/lava/darkimage/di_seismolava64__1.fits')
        # filterChain.setOpenBeamFitsFile('C:/Dev/huge data/Tomography/lava/openbeam/ob_seismolava64__1.fits')

        # filters = [ NoopFilter2d(), NoopFilter2d(), NoopFilter2d() ]
        # filterChain.setFilters(filters)

        # self.controlsLayout.append(filter.getControl()) for filter in filters

        imagewidget = ImagePlot(self)
        self.setCentralWidget(imagewidget)

        ui.checkColor.toggled.connect(outputChain.setColor)
        ui.checkInvert.toggled.connect(outputChain.setInvert)
        # ui.checkNormalize.toggled.connect(outputChain.setNormalize)
        ui.checkLogarithmic.toggled.connect(outputChain.setLog)

        ui.checkGrid.toggled.connect(imagewidget.setGridEnabled)

        def prepareCallback(input):
            print input
            return input

        def getDataCallback(input):
            print input
            return input

        customFilter = InvokeFilter()
        customFilter.setPredecessors([FilterVar(npfilter.filter())])
        customFilter.setTarget(prepareCallback, getDataCallback)

        fix2d = FixDimFilter2d()
        fix2d.setPredecessor(customFilter.filter())

        outputChain.setSource(fix2d.filter())

        validator.add(outputChain.statistic().validatable())
        validator.add(outputChain.image().validatable())

        self.validator = validator

        def validationCallback():
            if (outputChain.image().isValid()):
                imagewidget.setImage(outputChain.image().image())

        validator.validationStep.connect(validationCallback)

        def testCallback():
            outputChain.setSource(fitsfilter.filter())

        ui.buttonTest.clicked.connect(testCallback)

        validator.start()

if __name__ == '__main__':

    app = QApplication(sys.argv[1:])
    window = MainWindow(None)
    window.resize(1000, 600)
    window.show()
    app.exec_()
