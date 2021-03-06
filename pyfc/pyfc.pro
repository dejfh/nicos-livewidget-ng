TEMPLATE = lib
TARGET = pyfilterchain
CONFIG += staticlib

include(../defaults.prf)
include(../openmp.prf)

SOURCES += \
    filterchain.cpp \
    pyfc.cpp

HEADERS += \
    numpyinput.h \
    invokefilter.h \
    numpy.h \
    pyfilter.h \
    shared_ptr.h

DISTFILES += \
    setup.py \
    module_pyfc.sip \
    filter.sip \
    fixdimfilter.sip \
    buffer.sip \
    datastatistic.sip \
    invokefilter.sip \
    imageoutput.sip \
    validator.sip \
    numpyinput.sip \
    demo_fc.py \
    link_debug.cmd

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

unix: INCLUDEPATH += /usr/include/python2.7
else: INCLUDEPATH += C:/msys64/mingw64/include/python2.7

FORMS += \
    demo_fc.ui
