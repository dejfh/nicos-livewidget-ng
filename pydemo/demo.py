#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Live widget next gen demonstration."""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QPrinter, QPrintDialog, QPainter
from PyQt4.uic import loadUi

import numpy

# Allow running this after "python setup.py build" without install
sys.path[0:0] = glob.glob('../pyfc/build/lib.*')
sys.path[0:0] = glob.glob('../pyfcfits/build/lib.*')
sys.path[0:0] = glob.glob('../pyipw/build/lib.*')
sys.path[0:0] = glob.glob('../pytomo/build/lib.*')

from nicos_filterchain import QtValidator, ImageOutputChain, Numpy2d, InvokeFilter, FixDimFilter2d, FilterVar
from nicos_filterchain_fits import Fits2d
from nicos_imageplot import ImagePlot, RangeSelectWidget
from nicos_tomography import Tomography
from math import sqrt, pow

import numpy as np

class MainWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        ui = loadUi('demo.ui', self)

        validator = QtValidator(self)
        self.validator = validator

        size = 512
        center = size / 2
        count = 512
        radius = 105
        radiusf = 99.7

        tomography = Tomography(size, count, center)
        self.tomography = tomography

        outputChain = ImageOutputChain()

        npfilter = Numpy2d()
        outputChain.setSource(npfilter)

        openBeam = np.ones(size)
        sinogram = np.ones((size, count))
        for x in range(0, radius, 1):
            rx = (x + .5) / radiusf
            if rx >= 1: break
            d = sqrt( 1.0 - (rx * rx) )
            i = pow(.5, d)
            print rx, i
            sinogram[center - 1 - x, 0:count] = i
            sinogram[center + x, 0:count] = i
        angles = np.linspace(0, 360, count)
        tomography.setOpenBeam(openBeam)
        tomography.appendSinogram(sinogram, angles)
        tomography.requestReconstruction()

        npfilter.setData(sinogram)

        self.originalSinogram = sinogram

        def selectOriginalAsSource():
            npfilter.setData(sinogram)

        ui.radioSourceOriginal.clicked.connect(selectOriginalAsSource)

        def forceNextStep():
            tomography.setForceSteps(ui.radioSourceOriginal.isChecked())

        ui.buttonForce.clicked.connect(forceNextStep)

        imagewidget = ImagePlot(self)
        self.imagewidget = imagewidget
        self.setCentralWidget(imagewidget)

        rangewidget = RangeSelectWidget(ui.scrollAreaWidgetContents)
        self.rangewidget = rangewidget
        ui.scrollAreaWidgetContents.layout().addWidget(rangewidget)

        ui.checkColor.toggled.connect(outputChain.setColor)
        ui.checkInvert.toggled.connect(outputChain.setInvert)
        ui.checkLogarithmic.toggled.connect(outputChain.setLog)

        ui.checkGrid.toggled.connect(imagewidget.setGridEnabled)

        def doTomoStep():
            if ui.buttonTest.isChecked():
                tomography.run(-1)
                self.requestFromTomo()
            else:
                tomography.stop()
                self.requestFromTomo()

        ui.buttonTest.clicked.connect(doTomoStep)

        def tomoStepDone():
            if tomography.reconstructionAvailable():
                self.reconstruction = tomography.getReconstruction()
                if ui.radioSourceReconstruction.isChecked():
                    npfilter.setData(self.reconstruction)
            if tomography.sinogramAvailable():
                self.sinogram = tomography.getSinogram()
                if ui.radioSourceSinogram.isChecked():
                    npfilter.setData(self.sinogram)
            if tomography.likelihoodAvailable():
                self.likelihood = tomography.getLikelihood()
                if ui.radioSourceLikelihood.isChecked():
                    npfilter.setData(self.likelihood)
            if tomography.gradientAvailable():
                self.gradient= tomography.getGradient()
                if ui.radioSourceGradient.isChecked():
                    npfilter.setData(self.gradient)
            self.requestFromTomo()


        tomography.stepDone.connect(tomoStepDone)

        def rangeSelectionChanged(a, b):
            outputChain.setColormapRange(a, b)

        rangewidget.rangeChanged.connect(rangeSelectionChanged)

        validator.add(outputChain.statistic())
        validator.add(outputChain.image())

        self.imageBuffer = outputChain.image()
        self.statisticBuffer = outputChain.statistic()

        validator.validationStep.connect(self.validationCallback)
        validator.validationComplete.connect(self.onValidationComplete)

        validator.start()
        tomography.run(1)

    def validationCallback(self):
        if (self.imageBuffer.isValid()):
            self.imagewidget.setImage(self.imageBuffer.image())
        if (self.statisticBuffer.isValid()):
            self.rangewidget.setStatistic(self.statisticBuffer.statistic())

    def onValidationComplete(self):
        self.requestFromTomo()

    def requestFromTomo(self):
        self.tomography.requestReconstruction()
        self.tomography.requestSinogram()
        self.tomography.requestLikelihood()
        self.tomography.requestGradient()

if __name__ == '__main__':

    app = QApplication(sys.argv[1:])
    window = MainWindow(None)
    window.resize(1000, 600)
    window.show()
    app.exec_()
