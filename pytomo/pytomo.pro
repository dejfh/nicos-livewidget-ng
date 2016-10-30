TEMPLATE = lib
TARGET = pytomo
CONFIG += staticlib
QT       += opengl

include(../defaults.prf)
include(../openmp.prf)

SOURCES += \
    tomography.cpp \
    pytomo.cpp

HEADERS += \
    tomography.h \
    numpy.h

DISTFILES += \
    setup.py \
    module_pytomo.sip \
    demo_tomo.py \
    link_debug.cmd

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

unix: INCLUDEPATH += /usr/include/python2.7
else: INCLUDEPATH += C:/msys64/mingw64/include/python2.7

FORMS += \
    demo_tomo.ui
