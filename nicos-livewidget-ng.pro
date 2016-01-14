#-------------------------------------------------
#
# Project created by QtCreator 2015-01-21T10:12:22
#
#-------------------------------------------------

QT       += core gui opengl
CONFIG += c++11
CONFIG += qwt

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

unix {
    include ( /usr/local/qwt-6.1.2/features/qwt.prf )
}

msvc {
    QMAKE_CXXFLAGS *= /openmp
    QMAKE_CXXFLAGS += -D _CRT_SECURE_NO_WARNINGS
    QMAKE_CFLAGS = -nologo -Zm2000 -Zc:wchar_t -FS
    QMAKE_CXXFLAGS = -nologo -Zm2000 -Zc:wchar_t -FS
}
gcc {
    QMAKE_CXXFLAGS *= -fopenmp
    QMAKE_LFLAGS *= -fopenmp
}

TARGET = nicos-livewidget-ng
TEMPLATE = app

SOURCES += main.cpp\
    signalingprogress.cpp \
    initdialog.cpp \
    initdialogsettings.cpp \
    tw/projectlocation.cpp \
    tw/projectlocationsettings.cpp \
    test1.cpp

HEADERS  += \
    safecast.h \
    signalingprogress.h \
    plot2ddatadialog.h \
    asyncprogress.h \
    variadic.h \
    helper/helper.h \
    helper/variadic.h \
    helper/array.h \
    helper/union.h \
    helper/dispatcher.h \
    helper/qt/dispatcher.h \
    helper/qt/vector.h \
    helper/threadsafe.h \
    initdialog.h \
    initdialogsettings.h \
    tw/projectlocation.h \
    tw/projectlocationsettings.h \
    helper/copyonwrite.h

RESOURCES += \
    resources.qrc

FORMS += \
    initdialog.ui

DISTFILES += \
    doxygen.dox

include(ndim/NDim.pri)
include(filter/Filter.pri)
include(ndimdata/NDimData.pri)
include(ipw/ImagePlotWidget.pri)
include(fits/FitsHelper.pri)
include(lw/LiveWidget.pri)
