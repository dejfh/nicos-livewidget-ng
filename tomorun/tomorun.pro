TEMPLATE = app
TARGET = tomorun
CONFIG += console
CONFIG -= app_bundle
QT += opengl

include(../defaults.prf)

LIBS += -ltomography
LIBS += -lfilterchain

include(../openmp.prf)
include(../cfitsio.prf)

FORMS += \
    tomowindow.ui

HEADERS += \
    tomowindow.h

SOURCES += \
    tomowindow.cpp \
    main.cpp
