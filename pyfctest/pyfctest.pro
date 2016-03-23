TEMPLATE = app
TARGET = pyfctest
QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

include(../defaults.prf)

LIBS *= -lpyimageplotwidget -lpyfilterchain -lfilterchain -lpython2.7 -lqwt

include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

unix: INCLUDEPATH += /usr/include/python2.7
else: INCLUDEPATH += C:/msys64/mingw64/include/python2.7
