TARGET = fctest
CONFIG += console
CONFIG -= app_bundle

CONFIG += blubb

TEMPLATE = app

include(../defaults.prf)

include(../fc/FilterChain.pri)

SOURCES += main.cpp \
	headeronly.cpp \
    createinstances.cpp \
    createinstancesauto.cpp
