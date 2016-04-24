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
sys.path[0:0] = glob.glob('../pyipw/build/lib.*')

from nicosfilterchain import FilterChain #, NoopFilter2d
from nicosimageplot import ImagePlot #, HistogramPlot, RangeSelectWidget

# from nicosfilterchain import __VERSION__ as nicosfilterchainversion
# from nicosimagewidget import __VERSION__ as nicosimagewidgetversion


class MainWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        ui = loadUi('demo.ui', self)

        filterChain = FilterChain(self)
        self.fc = filterChain

        data1 = numpy.identity(3)
        data1 = numpy.array(data1, dtype='f')

        di = [[1,1,1],[1,1,1],[1,5,1]]

        ob = [[5,5,5],[9,9,9],[5,9,5]]

        im = [[5,3,4],[1,5,7],[5,7,4]]

        filterChain.setInput(im)
        filterChain.setDarkImage(di)
        filterChain.setOpenBeam(ob)

        # filterChain.setInputFitsFile('/home/felix/Projekte/daten/lava/raw/seismolava64__000.000.fits')

        # filterChain.setInputFitsFile('C:/Dev/huge data/Tomography/lava/raw/seismolava64__000.000.fits')
        # filterChain.setDarkImageFitsFile('C:/Dev/huge data/Tomography/lava/darkimage/di_seismolava64__1.fits')
        # filterChain.setOpenBeamFitsFile('C:/Dev/huge data/Tomography/lava/openbeam/ob_seismolava64__1.fits')

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

        def callback():
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            r = dialog.exec_()
            if r != 1: return
            painter = QPainter()
            painter.begin(printer)
            # painter.scale(3,3)
            self.imagewidget.render(painter)
            painter.end()

        ui.btnPrint.clicked.connect(callback)

if __name__ == '__main__':

    app = QApplication(sys.argv[1:])
    window = MainWindow(None)
    window.resize(1000, 600)
    window.show()
    app.exec_()
