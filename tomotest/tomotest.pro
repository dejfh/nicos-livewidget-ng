TEMPLATE = app
TARGET = tomotest
CONFIG += console
CONFIG -= app_bundle
QT += opengl

LIBS *= -ltomography
LIBS *= -lfilterchain

include(../defaults.prf)
include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += main.cpp \
	headeronly.cpp \
    createinstances.cpp \
    createinstancesauto.cpp
