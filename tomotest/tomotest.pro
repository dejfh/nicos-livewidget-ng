TEMPLATE = app
TARGET = tomotest
CONFIG += console
CONFIG -= app_bundle
QT += opengl

include(../defaults.prf)

LIBS += -ltomography
LIBS += -lfilterchain

include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += main.cpp \
	headeronly.cpp \
    createinstances.cpp \
    createinstancesauto.cpp
