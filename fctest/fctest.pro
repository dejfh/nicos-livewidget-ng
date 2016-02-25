TEMPLATE = app
TARGET = fctest
CONFIG += console
CONFIG -= app_bundle

LIBS *= -lfilterchain

include(../defaults.prf)
include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += main.cpp \
	headeronly.cpp \
    createinstances.cpp \
    createinstancesauto.cpp
