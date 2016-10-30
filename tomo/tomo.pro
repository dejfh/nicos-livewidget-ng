TEMPLATE = lib
TARGET = tomography
CONFIG += staticlib
QT       += opengl

include(../defaults.prf)
include(../openmp.prf)

HEADERS += \
    $$PWD/axisofrotation.h \
    $$PWD/findaxisofrotation.h \
    $$PWD/reconstructor.h \
    $$PWD/reconstructorerrors.h \
    $$PWD/shaders.h \
    $$PWD/sinogramfile.h \
    $$PWD/sinogramfileheader.h \
    $$PWD/threadableglwidget.h \
    tomography.h

SOURCES += \
    $$PWD/findaxisofrotation.cpp \
    $$PWD/reconstructor.cpp \
    $$PWD/shaders.cpp \
    $$PWD/sinogramfile.cpp \
    $$PWD/threadableglwidget.cpp \
    tomography.cpp
