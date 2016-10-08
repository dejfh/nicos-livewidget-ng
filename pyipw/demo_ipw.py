#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Plot Demo."""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QFileDialog
from PyQt4.uic import loadUi

sys.path[0:0] = glob.glob('../pyfc/build/lib.*')
sys.path[0:0] = glob.glob('../pyfcfits/build/lib.*')

sys.path[0:0] = glob.glob('./build/lib.*')

from nicos_imageplot import ImagePlot

class DemoWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        ui = loadUi('demo_ipw.ui', self)

        imagePlot = ImagePlot(None)
        ui.mainLayout.addWidget(imagePlot, 1)
        self.imagePlot = imagePlot

        ui.btnOpen.clicked.connect(self.openImage)
        ui.chkGrid.toggled.connect(imagePlot.setGridEnabled)

    def openImage(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg *.gif)")
        if not filename.isEmpty():
            self.imagePlot.setPixmap(QPixmap(filename))

if __name__ == '__main__':
    app = QApplication(sys.argv[1:])
    widget = DemoWidget(None)
    widget.resize(1000, 600)
    widget.show()
    app.exec_()
