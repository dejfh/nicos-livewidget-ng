#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FilterChain Demo"""

import sys
import glob

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QPixmap, QPrinter, QPrintDialog, QPainter
from PyQt4.uic import loadUi

import numpy as np

# Allow running this after "python setup.py build"
sys.path[0:0] = glob.glob('../pyfc/build/lib.*')

from nicos_filterchain import QtValidator, Numpy1d, Buffer1d, QtWatcher

class DemoWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        ui = loadUi('demo_fc.ui', self)

        self.inc = 1
        array = np.arange(10, 10 + 5 * self.inc, 5)
        numpy1d = Numpy1d()
        numpy1d.setData(array)
        self.numpy1d = numpy1d
        buffer = Buffer1d()
        buffer.setPredecessor(numpy1d)
        self.buffer = buffer

        validator = QtValidator(self)
        validator.add(buffer)
        self.validator = validator

        validator.validationProcCalled.connect(self.onValidationStarted)

        watcher = QtWatcher(validator, buffer, self)

        watcher.validated.connect(self.onValidated)
        watcher.invalidated.connect(self.onInvalidated)
        self.watcher = watcher

        ui.btnOpen.clicked.connect(self.openImage)

        validator.start()

    def openImage(self):
        self.inc += 1
        array = np.arange(10, 10 + 5 * self.inc, 5)
        self.numpy1d.setData(array)

    def onValidated(self):
        isValid = self.buffer.isValid()
        print "validated: ", isValid

    def onInvalidated(self):
        isValid = self.buffer.isValid()
        print "invalidated: ", isValid

    def onValidationStarted(self):
        print "validation started..."

if __name__ == '__main__':
    app = QApplication(sys.argv[1:])
    widget = DemoWidget(None)
    widget.resize(500, 200)
    widget.show()
    app.exec_()
