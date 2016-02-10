TEMPLATE = lib
TARGET = pyfilterchain
CONFIG += staticlib

include(../defaults.prf)
include(../openmp.prf)
include(../fc/FilterChain.pri)

SOURCES += \
    outputfilter.cpp \
    outputpixmap.cpp

HEADERS += filter2d.h \
    skipable2d.h \
    outputfilter.h \
    outputpixmap.h

DISTFILES +=

unix {
    target.path = /usr/lib
    INSTALLS += target
}
