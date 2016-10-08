#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Plot Demo."""

import sys
import glob

from math import exp
from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QFileDialog
from PyQt4.uic import loadUi

sys.path[0:0] = glob.glob('../pyfc/build/lib.*')
sys.path[0:0] = glob.glob('../pyfcfits/build/lib.*')

sys.path[0:0] = glob.glob('./build/lib.*')

from nicos_imageplot import RangeSelectWidget

from nicos_filterchain import DataStatistic

if __name__ == '__main__':
    app = QApplication(sys.argv[1:])
    widget = RangeSelectWidget(None)
    stat = DataStatistic()

    xs = range(-100, 100)
    list = [1000 * exp(-(x*x)/2500.0) for x in xs]
    stat.setHistogram(list)
    stat.setMin(-10)
    stat.setMax(10)
    stat.setAutoLowBound(-5)
    stat.setAutoHighBound(5)

    widget.setStatistic(stat)
    widget.setWindowTitle("Range Select Demo")

    widget.resize(500, 600)
    widget.show()
    app.exec_()
