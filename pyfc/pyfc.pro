TEMPLATE = lib
TARGET = pyfilterchain
CONFIG += shared

LIBS *= -lfilterchain -lpython2.7

include(../defaults.prf)
include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += \
    filterchain.cpp \
    pyfc.cpp

HEADERS += filter2d.h \
    skipable2d.h \
    pyfc.h \
    filterchain.h \
    numpyinput.h \
    noopfilter2d.h

DISTFILES += \
    pyfc.sip \
    setup.py

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

INCLUDEPATH += C:/msys64/mingw64/include/python2.7
