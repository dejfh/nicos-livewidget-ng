TEMPLATE = lib
TARGET = pyfilterchain
CONFIG += staticlib

include(../defaults.prf)

# LIBS += -lfilterchain -lpython2.7

include(../openmp.prf)

LIBS += -lcfitsio

SOURCES += \
    filterchain.cpp \
    pyfc.cpp

HEADERS += \
    pyfc.h \
    filterchain.h \
    numpyinput.h \
    filter2d.h

DISTFILES += \
    pyfc.sip \
    setup.py

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

unix: INCLUDEPATH += /usr/include/python2.7
else: INCLUDEPATH += C:/msys64/mingw64/include/python2.7
