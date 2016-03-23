TEMPLATE = app
TARGET = fctest
CONFIG += console
CONFIG -= app_bundle

include(../defaults.prf)

LIBS += -lfilterchain

include(../openmp.prf)
include(../cfitsio.prf)

SOURCES += main.cpp \
	headeronly.cpp \
    createinstances.cpp \
    createinstancesauto.cpp
