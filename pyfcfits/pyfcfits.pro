TEMPLATE = lib
TARGET = pyfilterchainfits
CONFIG += staticlib

include(../defaults.prf)

include(../openmp.prf)

LIBS += -lcfitsio

SOURCES +=

HEADERS +=

DISTFILES += \
    setup.py \
    module_pyfcfits.sip

setup.files = $$DISTFILES
setup.path = $$OUT_PWD

INSTALLS += setup

unix: INCLUDEPATH += /usr/include/python2.7
else: INCLUDEPATH += C:/msys64/mingw64/include/python2.7
