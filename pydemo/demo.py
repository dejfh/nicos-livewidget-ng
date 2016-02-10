#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Live widget next gen demonstration."""

import sys
import glob

from PyQt4.QtGui import QMainWindow, QApplication
from PyQt4.uic import loadUi

from nicosfilterchain import Validator, Input2dNumpy, Output2dNumpy, OutputPixmap, Filter2d

from nicosimagewidget import ImagePlotWidget

from nicosfilterchain import __version__ as nicosfilterchainversion
from nicosimagewidget import __version__ as nicosimagewidgetversion

# Allow running this after "python setup.py build"
sys.path[0:0] = glob.glob('build/lib.*')

class MainWindow(QMainWindow):
    def __init__(self, parent):
	QMainWindow.__init__(self, parent)
	loadUi('demo.ui', self)

	self.imagewidget = ImagePlotWidget(self)
	self.plotLayout.addWidget(self.imagewidget)

	x = open("testdata/testdata.npy").read()[80:]
	data = LWData(1024, 1024, 1, "<u4", x)

    def setWindowTitle(self, title):
	title += ' version:' + __version__
	QMainWindow.setWindowTitle(self, title)

    def onPixmap(self, pixmap):
	self.imagewidget.


if __name__ == '__main__':
    app = QApplication(sys.argv[1:])
    window = MainWindow(None)
    window.setWindowTitle('LiveWidget demo')
    window.resize(1000, 600)
    window.show()
    app.exec_()
