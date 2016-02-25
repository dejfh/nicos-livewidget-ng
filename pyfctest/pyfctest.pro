TEMPLATE = app
TARGET = pyfctest
QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

LIBS += -lpyfilterchain -lpyimageplotwidget

include(../defaults.prf)
include(../openmp.prf)

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += C:/msys64/mingw64/include/python2.7
