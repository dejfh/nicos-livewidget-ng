#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Live widget next gen demonstration."""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap
from PyQt4.uic import loadUi

import numpy

from nicosfilterchain import FilterChain #, NoopFilter2d
from nicosimageplot import ImagePlot #, HistogramPlot, RangeSelectWidget

# from nicosfilterchain import __VERSION__ as nicosfilterchainversion
# from nicosimagewidget import __VERSION__ as nicosimagewidgetversion

# Allow running this after "python setup.py build" ???
sys.path[0:0] = glob.glob('build/lib.*')

class MainWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        ui = loadUi('demo.ui', self)

        filterChain = FilterChain(self)
        self.fc = filterChain

        data = numpy.identity(200)
        data = numpy.array(data, dtype='f')
        print data
        # filterChain.setInput(data)

        # filterChain.setInputFitsFile('/home/felix/Projekte/daten/lava/raw/seismolava64__000.000.fits')

        filterChain.setInputFitsFile('C:/Dev/huge data/Tomography/lava/raw/seismolava64__000.000.fits')
        filterChain.setDarkImageFitsFile('C:/Dev/huge data/Tomography/lava/darkimage/di_seismolava64__1.fits')
        filterChain.setOpenBeamFitsFile('C:/Dev/huge data/Tomography/lava/openbeam/ob_seismolava64__1.fits')

        # filters = [ NoopFilter2d(), NoopFilter2d(), NoopFilter2d() ]
        # filterChain.setFilters(filters)

        # self.controlsLayout.append(filter.getControl()) for filter in filters

        self.imagewidget = ImagePlot(self)
        self.setCentralWidget(self.imagewidget)

        ui.checkColor.toggled.connect(filterChain.setUseColor)
        ui.checkInvert.toggled.connect(filterChain.setInvert)
        ui.checkNormalize.toggled.connect(filterChain.setNormalize)
        ui.checkLogarithmic.toggled.connect(filterChain.setLogarithmic)

        filterChain.pixmapChanged.connect(self.imagewidget.setImage)

        filterChain.start()

if __name__ == '__main__':

    app = QApplication(sys.argv[1:])
    window = MainWindow(None)
    window.resize(1000, 600)
    window.show()
    app.exec_()
