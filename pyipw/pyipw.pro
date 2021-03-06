TEMPLATE = lib
TARGET = pyimageplotwidget
CONFIG += staticlib

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

include(../defaults.prf)
include(../ipw/ImagePlotWidget.pri)

SOURCES +=

HEADERS += \
    pyipw.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

DISTFILES += \
    setup.py \
    module_pyipw.sip \
    demo_ipw.py \
    demo_range.py

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

FORMS += \
    demo_ipw.ui
