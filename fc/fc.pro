TEMPLATE = lib
TARGET = filterchain
CONFIG += staticlib
#QT       -= core gui

include(../defaults.prf)

SOURCES += \
    validation/qtvalidator.cpp

HEADERS += \
    filter/forward.h \
    filter/correction.h
unix {
    target.path = /usr/lib
    INSTALLS += target
}

include(../openmp.prf)

include(../fits/FitsHelper.pri)
include(../ndimdata/NDimData.pri)

HEADERS += \
    $$PWD/filter.h \
    $$PWD/filterbase.h \
    $$PWD/validation/validator.h \
    $$PWD/validation/watcher.h \
    $$PWD/gethelper.h \
    $$PWD/ndimcontainer.h \
    $$PWD/chains/fitspile.h \
    $$PWD/chains/darkimageopenbeam.h \
    $$PWD/chains/pixmapoutput.h \
    $$PWD/chains/profileplot.h \
    $$PWD/chains/zplot.h \
    $$PWD/chains/sourceselect.h \
    $$PWD/chains/dataprocess.h \
    $$PWD/datafilterbase.h \
    $$PWD/filter/perelement.h \
    $$PWD/filter/buffer.h \
    $$PWD/filter/switch.h \
    $$PWD/filter/accumulate.h \
    $$PWD/filter/extend.h \
    $$PWD/filter/fits.h \
    $$PWD/datafilter.h \
    $$PWD/filter/median.h \
    $$PWD/filter/analyzer.h \
    $$PWD/filter/pile.h \
    $$PWD/filter/profile.h \
    $$PWD/filter/subrange.h \
    $$PWD/filter/pixmap.h \
    $$PWD/filter/valuerange.h \
    $$PWD/filter/mapmerge.h \
    $$PWD/filter/input.h \
    $$PWD/validation/qtvalidator.h \
    $$PWD/validation/qtwatcher.h

SOURCES += \
    $$PWD/validation/validator.cpp \
    $$PWD/validation/watcher.cpp \
    $$PWD/chains/darkimageopenbeam.cpp \
    $$PWD/validation/qtwatcher.cpp

DISTFILES += \
    $$PWD/filter.dox
