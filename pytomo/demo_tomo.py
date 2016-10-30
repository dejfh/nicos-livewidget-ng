#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tomography Demo"""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QPrinter, QPrintDialog, QPainter
from PyQt4.uic import loadUi

import numpy as np

# Allow running this after "python setup.py build"
sys.path[0:0] = glob.glob('../pytomo/build/lib.*')

from nicos_tomography import Tomography

class DemoWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        ui = loadUi('demo_tomo.ui', self)

        self.tomography = Tomography(1024, 512, 0.5)

        ui.btnOpen.clicked.connect(self.openImage)
        ui.btnEnd.clicked.connect(self.getReconstruction)

    def openImage(self):
        openBeam = np.ones(1024)
        sinogram = np.ones((1024,200))
        sinogram[256:768,0:200]=0.5
        angles = np.linspace(0, 1, 200)
        self.tomography.setOpenBeam(openBeam)
        self.tomography.appendSinogram(sinogram, angles)
        self.tomography.run(5)
        self.tomography.requestReconstruction()

    def getReconstruction(self):
        data = self.tomography.getReconstruction()
        print data
        print data[512,512]

if __name__ == '__main__':
    app = QApplication(sys.argv[1:])
    widget = DemoWidget(None)
    widget.resize(500, 200)
    widget.show()
    app.exec_()
